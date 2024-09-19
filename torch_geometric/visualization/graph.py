from math import sqrt
from typing import Any, List, Optional
import numpy as np

import torch
from torch import Tensor

BACKENDS = {'graphviz', 'networkx'}


def has_graphviz() -> bool:
    try:
        import graphviz
    except ImportError:
        return False

    try:
        graphviz.Digraph().pipe()
    except graphviz.backend.ExecutableNotFound:
        return False

    return True


def visualize_graph(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    target: Optional[List[int]] = None,
    target_nodes: Optional[List[int]] = None,
    target_color: Optional[List[str]] = None,
    path: Optional[str] = None,
    backend: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    r"""Visualizes the graph given via :obj:`edge_index` and (optional)
    :obj:`edge_weight`.

    Args:
        edge_index (torch.Tensor): The edge indices.
        edge_weight (torch.Tensor, optional): The edge weights.
        path (str, optional): The path to where the plot is saved.
            If set to :obj:`None`, will visualize the plot on-the-fly.
            (default: :obj:`None`)
        backend (str, optional): The graph drawing backend to use for
            visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
            If set to :obj:`None`, will use the most appropriate
            visualization backend based on available system packages.
            (default: :obj:`None`)
        node_labels (List[str], optional): The labels/IDs of nodes.
            (default: :obj:`None`)
    """
    if edge_weight is not None:  # Normalize edge weights.
        edge_weight = edge_weight - edge_weight.min()
        edge_weight = edge_weight / edge_weight.max()

    if edge_weight is not None:  # Discard any edges with zero edge weight:
        mask = edge_weight > 1e-7
        edge_index = edge_index[:, mask]
        edge_weight = edge_weight[mask]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    if backend is None:
        backend = 'graphviz' if has_graphviz() else 'networkx'

    if backend.lower() == 'networkx':
        return _visualize_graph_via_networkx(edge_index, edge_weight, target, target_nodes, target_color, path,
                                             node_labels)
    elif backend.lower() == 'graphviz':
        return _visualize_graph_via_graphviz(edge_index, edge_weight, path,
                                             node_labels)

    raise ValueError(f"Expected graph drawing backend to be in "
                     f"{BACKENDS} (got '{backend}')")


def _visualize_graph_via_graphviz(
    edge_index: Tensor,
    edge_weight: Tensor,
    path: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    import graphviz

    suffix = path.split('.')[-1] if path is not None else None
    g = graphviz.Digraph('graph', format=suffix)
    g.attr('node', shape='circle', fontsize='11pt')

    for node in edge_index.view(-1).unique().tolist():
        g.node(str(node) if node_labels is None else node_labels[node])

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        hex_color = hex(255 - round(255 * w))[2:]
        hex_color = f'{hex_color}0' if len(hex_color) == 1 else hex_color
        if node_labels is not None:
            src = node_labels[src]
            dst = node_labels[dst]
        g.edge(str(src), str(dst), color=f'#{hex_color}{hex_color}{hex_color}')

    if path is not None:
        path = '.'.join(path.split('.')[:-1])
        g.render(path, cleanup=True)
    else:
        g.view()

    return g


def _visualize_graph_via_networkx(
    edge_index: Tensor,
    edge_weight: Tensor,
    targets: Optional[List[int]] = None,
    target_nodes: Optional[List[int]] = None,
    target_colors: Optional[List[str]] = None,
    path: Optional[str] = None,
    node_labels: Optional[List[str]] = None,
) -> Any:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    g = nx.DiGraph()
    node_size = 600

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node if node_labels is None else node_labels[node])

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        if node_labels is not None:
            src = node_labels[src]
            dst = node_labels[dst]
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    pos = nx.spring_layout(g)
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="->",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ),
        )
    target_to_color = {t:target_colors[t] for t in targets}

    if target_colors is not None:
        tn = target_nodes.cpu().numpy()
        colors = [target_colors[targets[np.where(tn==node)[0].item()].cpu().numpy()] for node in g.nodes()]
    else:
        colors = ['white']*len(g.nodes())
    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size,
                                   node_color=colors, margins=0.1)
    # ax = create_legend(target_to_color)
    nodes.set_edgecolor('black')
    nx.draw_networkx_labels(g, pos, font_size=10)

    if target_colors is not None and targets is not None:
        # Unique targets and corresponding colors
        unique_targets = list(set(targets.cpu().numpy()))
        target_to_color = {t: target_colors[t] for t in unique_targets}
        
        # Create custom legend handles
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Target {t}',
                                  markerfacecolor=target_to_color[t], markersize=10, markeredgecolor='black')
                           for t in unique_targets]

        # Add the legend to the plot
        ax.legend(handles=legend_elements, loc='upper right')

    if path is not None:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

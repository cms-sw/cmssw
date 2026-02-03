#!/usr/bin/env python3

from collections import defaultdict, deque
from typing import Dict, Set, List


import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *


def flatten_all_to_module_list(process, user_args):
    module_list = []
    for name in user_args:
        if not hasattr(process, name):
            print(f"[WARN] process has no attribute named '{name}'")
            continue
        obj_ = getattr(process, name)
        if hasattr(obj_, "moduleNames"):
            print("is sequence")
            module_list.extend(obj_.moduleNames())
        else:
            module_list.append(name)
    return module_list


def extract_inputtags(value):
    """
    Recursively extract cms.InputTag objects from a parameter value.
    Returns a list of cms.InputTag.
    """
    tags = []

    if isinstance(value, cms.InputTag):
        tags.append(value)

    elif isinstance(value, cms.VInputTag):
        for elem in value:
            if isinstance(elem, cms.InputTag):
                tags.append(elem)
            elif isinstance(elem, str):
                tags.append(cms.InputTag(elem))

    elif isinstance(value, cms.PSet):
        for v in value.parameters_().values():
            tags.extend(extract_inputtags(v))

    elif isinstance(value, (list, tuple)):
        for v in value:
            tags.extend(extract_inputtags(v))

    return tags

def get_module_dependencies(process, module_names):
    """
    For each module in module_names, print all producer modules it directly depends on via InputTag.
    Return set of all dependencies for the given modules
    """

    deps = set()

    for name in module_names:
        if not hasattr(process, name):
            print(f"[WARN] process has no module named '{name}'")
            continue

        module = getattr(process, name)

        for param in module.parameters_().values():
            for tag in extract_inputtags(param):
                producer = tag.getModuleLabel()
                if producer:
                    deps.add(producer)
                    print(f"\n{name} ({module.type_()}):")
                    print(f"  depends on: {producer}")

    return deps


def build_restricted_dependency_graph(process, modules: Set[str]) -> Dict[str, Set[str]]:
    """
    Build a directed graph A -> B where B depends on A,
    restricted to the given set of modules.
    """

    graph: Dict[str, Set[str]] = defaultdict(set)

    # Ensure all nodes exist
    for m in modules:
        graph.setdefault(m, set())

    for consumer in modules:
        if not hasattr(process, consumer):
            continue

        mod = getattr(process, consumer)

        for param in mod.parameters_().values():
            for tag in extract_inputtags(param):
                producer = tag.getModuleLabel()
                if producer in modules:
                    graph[producer].add(consumer)
    
    return graph



def connected_groups(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """
    Return list of weakly connected components.
    Each component is returned as a list of modules ordered
    by dependency from root to leaf.
    """

    # ---- build undirected graph ----
    undirected = defaultdict(set)

    for src, dsts in graph.items():
        undirected[src]  # ensure node exists
        for dst in dsts:
            undirected[src].add(dst)
            undirected[dst].add(src)

    seen = set()
    components = []

    # ---- find weakly connected components ----
    for node in undirected:
        if node in seen:
            continue

        stack = [node]
        comp = set()

        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            comp.add(n)
            stack.extend(undirected[n] - seen)

        components.append(comp)

    # ---- order each component by dependencies ----
    ordered_components = []

    for comp in components:
        # compute in-degree restricted to this component
        indegree = {n: 0 for n in comp}
        local_edges = defaultdict(set)

        for src in comp:
            for dst in graph.get(src, []):
                if dst in comp:
                    local_edges[src].add(dst)
                    indegree[dst] += 1

        # Kahn's algorithm
        queue = deque(sorted(n for n in comp if indegree[n] == 0))
        ordered = []

        while queue:
            n = queue.popleft()
            ordered.append(n)
            for dst in local_edges.get(n, []):
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    queue.append(dst)

        # If there is a cycle, append remaining nodes deterministically
        if len(ordered) < len(comp):
            remaining = sorted(comp - set(ordered))
            ordered.extend(remaining)

        ordered_components.append(ordered)

    return ordered_components



def print_dependency_groups(graph: Dict[str, Set[str]]):
    groups = connected_groups(graph)

    for i, group in enumerate(groups, 1):
        print(f"\nDependency group {i}:")

        for src in sorted(group):
            for dst in sorted(graph.get(src, [])):
                if dst in group:
                    print(f"  {src} -> {dst}")

        # Handle isolated nodes
        isolated = [
            m for m in group
            if not graph.get(m) and all(m not in v for v in graph.values())
        ]
        for m in isolated:
            print(f"  {m} (isolated)")


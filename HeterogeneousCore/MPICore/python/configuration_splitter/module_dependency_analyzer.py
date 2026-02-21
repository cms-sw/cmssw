from collections import defaultdict, deque
from typing import Dict, Set, List

import FWCore.ParameterSet.Config as cms


def flatten_all_to_module_set(process, user_args):
    """
    This function ensures that if one of the input arguments was path or sequence, 
    it will be flattened to a module set for input consistency
    """
    module_list = set()
    for name in user_args:
        if not hasattr(process, name):
            print(f"[WARN] process has no attribute named '{name}'")
            continue
        obj_ = getattr(process, name)
        if hasattr(obj_, "moduleNames"):
            print("is sequence")
            module_list.extend(obj_.moduleNames())
        else:
            module_list.add(name)
    return module_list


class ModuleDependencyAnalyzer:
    def __init__(self, process):
        self.process = process

        # ---- cached core structures ----
        self.module_inputs: Dict[str, Set[str]] = defaultdict(set)
        self.producer_to_consumers: Dict[str, Set[str]] = defaultdict(set)

        self._build_dependency_maps()


    def _build_dependency_maps(self):
        """
        Core extraction of dependencies based on input tags (DONE ONCE)
        """
        for name in self.process.producers_():
            mod = getattr(self.process, name)

            for param in mod.parameters_().values():
                for tag in self._extract_inputtags(param):
                    producer = tag.getModuleLabel()
                    if producer:
                        self.module_inputs[name].add(producer)
                        self.producer_to_consumers[producer].add(name)

    def _extract_inputtags(self, value):
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
                tags.extend(self._extract_inputtags(v))

        elif isinstance(value, (list, tuple)):
            for v in value:
                tags.extend(self._extract_inputtags(v))

        return tags



    def direct_dependencies(self, modules: Set[str]) -> Set[str]:
        """
        Get the modules whose products are needed
        """
        deps = set()
        for m in modules:
            deps |= self.module_inputs.get(m, set())
        return deps

    def _consumers_of(self, producer: str) -> Set[str]:
        """
        Get the modules which need the products of producer
        """
        return self.producer_to_consumers.get(producer, set())


    def _restricted_graph(self, modules: Set[str]) -> Dict[str, Set[str]]:
        """
        Restricted dependency graph, reflecting the relationships between the modules to offload
        """
        graph = defaultdict(set)
        for m in modules:
            graph.setdefault(m, set())

        for consumer in modules:
            for producer in self.module_inputs.get(consumer, []):
                if producer in modules:
                    graph[producer].add(consumer)

        return graph
    

    def _connected_groups(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
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


    def dependency_groups(self, modules: Set[str]) -> List[List[str]]:
        """
        Get dependency groups (ordered)
        """
        graph = self._restricted_graph(modules)
        return self._connected_groups(graph)



    def grouped_external_dependencies(
        self,
        groups: List[List[str]],
    ) -> List[Set[str]]:
        """
        Grouped external dependencies
        """

        grouped = []
        for group in groups:
            gset = set(group)
            deps = set()

            for m in group:
                for prod in self.module_inputs.get(m, []):
                    if prod not in gset:
                        deps.add(prod)

            grouped.append(deps)

        return grouped


    def producer_to_groups(
        self,
        grouped_deps: List[Set[str]],
    ) -> Dict[str, Set[int]]:
        """
        Producer → groups map
        """

        mapping = defaultdict(set)
        for gi, deps in enumerate(grouped_deps):
            for prod in deps:
                mapping[prod].add(gi)
        return mapping


    def modules_to_send_back_by_group(
        self,
        groups: List[List[str]],
        modules_to_run_on_both: Set[str],
    ):
        """
        Which offloaded modules must send products back, and which are not needed on local
        """
        module_to_group = {
            m: gi for gi, g in enumerate(groups) for m in g
        }

        result = [[] for _ in groups]
        unused = []

        for gi, group in enumerate(groups):
            for produced in group:
                if produced in modules_to_run_on_both:
                    continue

                needed = False
                for consumer in self._consumers_of(produced):
                    if module_to_group.get(consumer) != gi:
                        needed = True
                        break

                if needed:
                    result[gi].append(produced)
                else:
                    unused.append(produced)

        return result, unused

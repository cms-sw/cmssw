from collections import defaultdict
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

# the origin of the data products of the Source, and of those read from the input
SOURCE = "source"


class ConsumedProduct(NamedTuple):
    """
    One data product a module consumes in the Event.
    """

    # the (label, instance, process) the module asks for; label may be an EDAlias
    tag: Tuple[str, str, str]
    # the module producing the data product, or "source" for the Source and for
    # everything read from the input file.
    origin: str
    # its product_names.json entry; for an EDAlias, the entry of the original
    # data product, found by _original().
    product: dict
    # the product instance under which the module finds the data product.
    instance: str


class ModuleDependencyAnalyzer:
    def __init__(self, graph, products):
        """
        graph is the JSON produced by the DumpDependencyGraph service, products
        are the entries produced by DumpProductNames, grouped by module label.
        """
        modules = graph["modules"]
        self.process_name = graph["process"]

        # the module dependencies, resolved by the framework
        self.module_inputs: Dict[str, Set[str]] = {
            label: set(module["consumes"]) for label, module in modules.items()
        }

        # the Event data products each module declares it consumes
        self._consumes_products = {label: module["consumesProducts"] for label, module in modules.items()}

        # the Event data products known to the process, by (module label, product instance)
        self._products: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        for entries in products.values():
            for product in entries:
                if product["branch"] == 0:
                    self._products[(product["module"], product["product_instance"])].append(product)

        # Run/lumi/process-block dependencies
        self._non_event_inputs: Dict[str, Set[str]] = {
            label: set(module["consumesNonEvent"]) for label, module in modules.items()
        }

        # the condition under which each module runs
        self._reachability = self._build_reachability(graph)

    def consumed_products(self, module: str) -> List[ConsumedProduct]:
        """
        For each declared input of a module, find the data products with the
        same label and instance, and with the same type (or, for an edm::View, a
        compatible type).
        """
        result = []
        for entry in self._consumes_products.get(module, ()):
            tag = (entry["label"], entry["instance"], entry["process"])
            # an edm::View consumer declares the element type, which matches the containers of that
            # type or of a type derived from it
            matches = [
                p
                for p in self._products.get(tag[:2], ())
                if (entry["type"] in p.get("view_types", ()) if entry["elementType"] else p["friendly_type_name"] == entry["type"])
            ]
            if tag[2] == "@skipCurrentProcess":
                matches = [p for p in matches if p["process"] != self.process_name]
            elif tag[2]:
                matches = [p for p in matches if p["process"] == tag[2]]
            else:
                matches = [p for p in matches if p["process"] == self.process_name] or matches

            for p in matches:
                if p["process"] != self.process_name:
                    # read from the input, which only the Source provides
                    result.append(ConsumedProduct(tag, SOURCE, p, p["product_instance"]))
                elif "alias_for" in p:
                    original = self._original(p)
                    result.append(ConsumedProduct(tag, original["module"], original, p["product_instance"]))
                else:
                    result.append(ConsumedProduct(tag, p["module"], p, p["product_instance"]))
        return result

    def _original(self, alias: dict) -> dict:
        """
        Follows the `alias_for` field of an EDAlias entry to the entry of the
        real data product.
        """
        target = alias["alias_for"]
        return next(
            p
            for p in self._products[(target["module"], target["product_instance"])]
            if p["friendly_type_name"] == alias["friendly_type_name"] and p["process"] == self.process_name
        )

    def non_event_dependencies(self, modules: List[str]) -> Dict[str, Set[str]]:
        """
        Find the Run/Lumi/ProcessBlock inputs of the offloaded modules that
        stay local. `split_remote()` prints a warning for each one, because
        nothing forwards such products.
        """
        offloaded = set(modules)

        # the run, lumi and process block producers the offloaded modules need but which stay local
        outside = {}
        for module in modules:
            # a producer that is offloaded too follows the module to the remote process
            producers = self._non_event_inputs.get(module, set()) - offloaded
            if producers:
                outside[module] = producers

        return outside

    @staticmethod
    def _build_reachability(graph) -> Dict[str, FrozenSet[Tuple[str, int]]]:
        """
        Compute, for each module, the positions in the schedule from which the
        module is reached. A "position" is a pair "path, run index". `run_index`
        starts at 0 and increases after each EDFilter on the path. Two modules
        with the same set of positions run under the same condition.
        """
        # the edm type of every module, e.g. "Source", "EDProducer", "EDFilter"
        types = {label: module["type"] for label, module in graph["modules"].items()}

        # for each module, the set of positions in the schedule from which it is reached
        reachability = defaultdict(set)
        for kind in ("paths", "endpaths"):
            for name, labels in graph[kind].items():
                run_index = 0
                for label in labels:
                    reachability[label].add((f"{kind}:{name}", run_index))

                    if types.get(label) == "EDFilter":
                        run_index += 1

        return {label: frozenset(runs) for label, runs in reachability.items()}

    def reachability_signature(self, module_name: str) -> Optional[FrozenSet[Tuple[str, int]]]:
        """
        Returns the set computed by _build_reachability for a module. Returns
        `None` for a module on no path.
        """
        return self._reachability.get(module_name)

    def dependency_groups(self, modules: List[str]) -> List[List[str]]:
        """
        Split the offloaded modules into groups, that are moved, gated, and sent
        back together: modules that depend on each other and that are reached
        under the same condition end up in one group.
        """

        # the offloaded modules
        selected = set(modules)

        # each module's inputs, restricted to the offloaded set
        inputs = {m: self.module_inputs.get(m, set()) & selected for m in modules}

        # the (component, condition) key that decides which group each module belongs to
        keys = self._grouping_keys(modules, inputs)

        # gather modules with the same key
        members: Dict[Tuple, List[str]] = {}
        for m in modules:
            members.setdefault(keys[m], []).append(m)

        # compute which keys consume from which other keys
        depends_on = {key: set() for key in members}
        for consumer, producers in inputs.items():
            for producer in producers:
                if keys[producer] != keys[consumer]:
                    depends_on[keys[consumer]].add(keys[producer])

        # merge back the jeys that consume from each other in both directions
        groups = []
        for component in self._strongly_connected_components(list(members), depends_on):
            merged = set(component)
            groups.append(self._in_dependency_order([m for m in modules if keys[m] in merged], inputs))

        return groups

    def _grouping_keys(self, modules: List[str], inputs: Dict[str, Set[str]]) -> Dict[str, Tuple]:
        """
        Two modules are in the same group only when they are connected through
        dependencies and run under the same condition. A module on no path has
        no signature. This method gives each module the key (connected
        component, reachability signature). Modules with the same key will be
        grouped together.
        """

        component = self._connected_components(modules, inputs)
        keys = {m: (component[m], self.reachability_signature(m)) for m in modules}

        consumers: Dict[str, List[str]] = defaultdict(list)
        for consumer, producers in inputs.items():
            for producer in producers:
                consumers[producer].append(consumer)

        unscheduled = [m for m in modules if keys[m][1] is None]
        while unscheduled:
            resolved = []
            for m in unscheduled:
                for neighbour in consumers.get(m, []) + sorted(inputs[m]):
                    if keys[neighbour][1] is not None:
                        keys[m] = keys[neighbour]
                        resolved.append(m)
                        break
            if not resolved:
                break
            unscheduled = [m for m in unscheduled if m not in resolved]

        return keys

    @staticmethod
    def _connected_components(modules: List[str], inputs: Dict[str, Set[str]]) -> Dict[str, str]:
        """
        A "component" is a set of offloaded modules connected to each other by
        dependencies, in either direction, possibly through other offloaded
        modules. For example, if B reads A and C reads B, then A, B and C form
        one component. A module with no dependency on another offloaded module
        forms a component on its own. Only the dependencies inside the
        offloaded set count: two modules that read the same local producer are
        not connected by that producer.

        Every consumer-producer pair merges the two components it connects.
        Returns, for each module, the name of its component.
        """
        parent = {m: m for m in modules}

        def root(module_name):
            """Climb to the top of the chain, flattening it on the way."""
            while parent[module_name] != module_name:
                parent[module_name] = parent[parent[module_name]]
                module_name = parent[module_name]
            return module_name

        for consumer, producers in inputs.items():
            for producer in producers:
                parent[root(producer)] = root(consumer)

        # name each module's component after the top-level one it ends up under
        return {m: root(m) for m in modules}

    @staticmethod
    def _strongly_connected_components(nodes: List, edges: Dict) -> List[List]:
        """
        Tarjan's algorithm: return the sets of nodes that all reach each other
        """

        index: Dict = {}
        lowlink: Dict = {}
        stack: List = []
        on_stack: Set = set()

        # the finished components, in the order they were closed
        components: List[List] = []

        for start in nodes:
            if start in index:
                continue

            index[start] = lowlink[start] = len(index)
            stack.append(start)
            on_stack.add(start)

            work = [(start, iter(edges.get(start, ())))]

            while work:
                # resume the deepest unfinished node, where its iterator of successors left off
                node, successors = work[-1]
                for successor in successors:
                    if successor not in index:
                        # descend into it, and come back to this node's iterator afterwards
                        index[successor] = lowlink[successor] = len(index)
                        stack.append(successor)
                        on_stack.add(successor)
                        work.append((successor, iter(edges.get(successor, ()))))
                        break
                    if successor in on_stack:
                        # a successor still on the stack loops back here, so it shares this component
                        lowlink[node] = min(lowlink[node], index[successor])
                else:
                    # a node whose lowlink never left its own visit number closes a component
                    if lowlink[node] == index[node]:
                        component = []
                        while True:
                            member = stack.pop()
                            on_stack.discard(member)
                            component.append(member)
                            if member == node:
                                break
                        components.append(component)
                    work.pop()

                    if work:
                        lowlink[work[-1][0]] = min(lowlink[work[-1][0]], lowlink[node])

        return components

    @staticmethod
    def _in_dependency_order(component: List[str], inputs: Dict[str, Set[str]]) -> List[str]:
        """
        Sort a group's modules so that each one comes after the modules it consumes from.
        """
        # build the order in passes, tracking what has been placed and what is left
        ordered: List[str] = []
        placed: Set[str] = set()
        remaining = component

        while remaining:
            ready = [m for m in remaining if inputs[m] <= placed]
            if not ready:
                return ordered + remaining
            ordered += ready
            placed.update(ready)
            remaining = [m for m in remaining if m not in placed]

        return ordered

    def external_dependencies_by_group(
        self, groups: List[List[str]]
    ) -> Tuple[Dict[str, Set[int]], Dict[str, List[ConsumedProduct]]]:
        """
        Determine what needs to go from local to remote.
        For every member of every group, take the consumed products whose origin
        (the module producing them) is not offloaded. Return two dicts, keyed by
        origin: one with the indices of the groups that need the origin, and one
        with the `ConsumedProduct`s read from the origin.
        """
        # every module that moves to the remote process, in any group
        offloaded = {m for group in groups for m in group}

        # a data product coming from outside that set is produced locally, so record who needs it
        groups_by_producer = defaultdict(set)
        consumed_by_producer = defaultdict(list)
        for group_index, group in enumerate(groups):
            for m in group:
                for consumed in self.consumed_products(m):
                    if consumed.origin not in offloaded:
                        groups_by_producer[consumed.origin].add(group_index)
                        consumed_by_producer[consumed.origin].append(consumed)
        return groups_by_producer, consumed_by_producer

    def modules_to_send_back_by_group(self, groups: List[List[str]], modules_to_run_on_both: List[str]):
        """
        Determine what needs to come back from remote to local, by looking at
        every module that stays local and collecting the data products it reads
        from remote-only modules.
        """

        # the modules that end up existing only remotely; a duplicated one runs
        # locally too
        remote_only = {m for group in groups for m in group} - set(modules_to_run_on_both)

        # what the modules that still run locally read from the remote-only modules
        read_locally = defaultdict(list)
        for consumer in self._consumes_products:
            if consumer not in remote_only:
                for consumed in self.consumed_products(consumer):
                    if consumed.origin in remote_only:
                        read_locally[consumed.origin].append(consumed)

        # per group, the members whose products a local module reads (and thus
        # need to be sent back).
        to_send = [[] for _ in groups]
        # per group, the exact consumed products to send back.
        consumed_by_group = [[] for _ in groups]

        # the members no local module reads, across all groups. A "member" here
        # is an offloaded module, as part of the group it was assigned to by
        # dependency_groups().
        unused = []

        # a member is worth sending back only if at least one of its data products is read locally
        for group_index, group in enumerate(groups):
            for m in group:
                if m in modules_to_run_on_both:
                    continue
                if read_locally[m]:
                    to_send[group_index].append(m)
                    consumed_by_group[group_index] += read_locally[m]
                else:
                    unused.append(m)

        return to_send, unused, consumed_by_group

"""
Methods that build the components of a split. These are called only by
split_remote.py:

  - the remote cms.Process, and the MPIController of the local process;
  - the MPISender/MPIReceiver pairs that move data products between the
    processes;
  - the EDAliases that make the moved data products visible under the labels
    the consumers use;
  - the PathStateCapture/PathStateRelease pairs that reproduce the schedule
    across the processes;
  - the new Paths.
"""

import itertools

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import insert_modules_before

from HeterogeneousCore.MPICore.modules import MPISource


def add_new(process, label, obj):
    """
    Add a module or path created by the splitter to the process, under `label`.
    A plain setattr() on a cms.Process silently replaces whatever already has
    that label, so a generated label that happens to match an existing one
    would replace a module of the configuration, or another generated module.
    This can happen because the generated labels are built with str.title():
    "hltFooBar" and "hltFoobar" both give "Hltfoobar". Raise an error instead.
    """
    if hasattr(process, label):
        raise RuntimeError(
            f"the splitter needs the label '{label}' in the process '{process.name_()}', but the label is already "
            "taken, either by the configuration or by another generated component (generated labels are built "
            "with str.title(), so module labels that differ only in case give the same generated label)"
        )
    setattr(process, label, obj)


class _TaskCollector:
    """Collects the Tasks of a sequence."""

    def __init__(self):
        self.tasks = []

    def enter(self, node):
        if isinstance(node, cms.Task):
            self.tasks.append(node)

    def leave(self, node):
        pass


def delete_module(process, label):
    """
    delattr() that also removes the module from the anonymous Tasks of the
    paths, which would otherwise be written as cms.Task(process.None).
    """
    module = getattr(process, label)
    delattr(process, label)

    collector = _TaskCollector()
    for sequence in itertools.chain(
        process.sequences_().values(), process.paths_().values(), process.endpaths_().values()
    ):
        sequence.visit(collector)
    for task in collector.tasks:
        task.remove(module)


def add_controller_to_local(process, remote_name):
    """
    Load the accelerator and MPI services into the local process, and add
    an `MPIController`. Also, set `numberOfConcurrentLuminosityBlocks = 1` until
    support is added for more than one concurrent luminosity block.
    Return the label of the controller.
    """
    process.load("Configuration.StandardSequences.Accelerators_cff")
    process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")
    controller_name = f"mpiController{remote_name.title()}"
    controller = cms.EDProducer("MPIController",
                               followerProcessName = cms.string(remote_name))
    add_new(process, controller_name, controller)
    # Multiple luminosity blocks are currently unsupported
    process.options.numberOfConcurrentLuminosityBlocks = 1
    return controller_name


def create_remote_process(local_process, modules_to_run, remote_process_name, local_process_name):
    """
    Build a new remote cms.Process. Copy from the local process the global
    PSets, the EventSetup sources and producers, and the MessageLogger. Load
    the MPI Services, set the MPISource as the source, and clone every
    offloaded module. EDAliases are handled later.
    """
    remote_process = cms.Process(remote_process_name)
    remote_process.load("Configuration.StandardSequences.Accelerators_cff")

    # load the global psets and event setup modules
    for module in local_process.psets.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())
    for module in local_process.es_sources.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())
    for module in local_process.es_producers.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())

    if hasattr(local_process, "MessageLogger"):
        remote_process.MessageLogger = local_process.MessageLogger.clone()

    remote_process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    remote_process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")

    remote_process.source = MPISource(
        mode = 'CommWorld',
        controllerProcessName = local_process_name
    )
    remote_process.maxEvents.input = -1

    for module in modules_to_run:
        if not hasattr(local_process, module):
            raise RuntimeError(f"Module {module} not found in source process")
        setattr(remote_process, module, getattr(local_process, module).clone())

    return remote_process



def is_device_product(prod):
    """Whether a product_names.json entry is a device product (edm::DeviceProduct<T>)."""
    return prod["type"].startswith("edm::DeviceProduct")


def _branch_label(module, product_instance):
    """
    The product instance name under which a receiver carrying several modules' products
    registers this one, so that modules producing the same instance name do not collide.
    """
    return f"{module}@{product_instance}" if product_instance else module


def _forwarded_products(products):
    """
    Every product an MPISender/MPIReceiver pair carries over MPI.
    """
    return [p for p in products if not is_device_product(p)]


def _branch_label_of(product):
    """_branch_label() of a product_names.json entry."""
    return _branch_label(product["module"], product["product_instance"])


def _sender_psets(products):
    """Write one PSet per data product, for the `products` parameter of an MPISender."""
    return [
        cms.PSet(
            type=cms.string(p["type"]),
            name=cms.InputTag(p["module"], p["product_instance"], p.get("sender_process", "")),
        )
        for p in _forwarded_products(products)
    ]


def _receiver_psets(products, grouped):
    """Write one PSet per data product, for the `products` parameter of an MPIReceiver."""
    return [
        cms.PSet(
            type=cms.string(p["type"]),
            label=cms.string(
                _branch_label(p["module"], p["product_instance"]) if grouped else p["product_instance"]
            ),
        )
        for p in _forwarded_products(products)
    ]


def create_sender(products, instance, upstream, activity=None):
    """ 
    Create an MPISender
    """
    parameters = dict(
        upstream=cms.InputTag(upstream),
        instance=cms.int32(instance),
        # write one PSet per data product
        products=cms.VPSet(*_sender_psets(products)),
    )
    if activity is not None:
        parameters["activity"] = cms.InputTag(activity)

    return cms.EDProducer("MPISender", **parameters)


def create_receiver(products, instance, upstream, activity=False, grouped=False):
    """
    Create an MPIReceiver.
    """
    return cms.EDProducer(
        "MPIReceiver",
        upstream=cms.InputTag(upstream),
        instance=cms.int32(instance),
        # write one PSet per data product
        products=cms.VPSet(*_receiver_psets(products, grouped)),
        activity=cms.bool(activity),
    )


def alias_entry(product, to_instance, grouped):
    """
    What an EDAlias needs to put `product` under the product instance `to_instance`: its
    friendly type, and the product instance it has in the module the EDAlias points to (a
    receiver registers it as modulelabel@instance when `grouped`).
    """
    from_instance = _branch_label_of(product) if grouped else product["product_instance"]
    return (product["friendly_type_name"], from_instance, to_instance)


def create_alias(entries):
    """
    Create an EDAlias.
    """

    return cms.EDAlias(
        **{
            module: cms.VPSet(
                *(
                    cms.PSet(
                        type=cms.string(type_name),
                        fromProductInstance=cms.string(from_instance),
                        toProductInstance=cms.string(to_instance),
                    )
                    for type_name, from_instance, to_instance in sorted(module_entries)
                )
            )
            for module, module_entries in sorted(entries.items())
        }
    )


# -- the activity filters --

def add_activity_filter(process, module_name, filter_name):
    """
    creates a PathStateRelease filter, that stops its path unless the token
    carried by `module_name`, (an MPIReceiver), arrived with the event.
    """
    filter_object =  cms.EDFilter("PathStateRelease",
            state = cms.InputTag(module_name)
        )
    add_new(process, filter_name, filter_object)


def insert_path_state_capture_before(
    process,
    first_modules_in_a_group,
    capture_name,
):
    """
    Create a PathStateCapture named `capture_name`, and insert the capture in front of each module of
    `first_modules_in_a_group`, in every Sequence, Path and EndPath of `process` that contains the
    module. When a path reaches one of these positions, the capture puts a token in the Event; when no
    path reaches any of them, the token is absent.
    """

    # create the PathStateCapture
    add_new(
        process,
        capture_name,
        cms.EDProducer("PathStateCapture"),
    )

    capture = getattr(process, capture_name)

    # insert into sequences
    for module_name in first_modules_in_a_group:
        if not hasattr(process, module_name):
            print(f"[WARN] process has no module '{module_name}'")
            continue

        module = getattr(process, module_name)

        insert_modules_before(process, module, capture)


# -- paths --

def make_new_path(
    process,
    path_name: str,
    module_names: list[str],
    ignored=(),
):
    """
    Create a cms.Path from an ordered list of module names
    and optionally append it to the process schedule. The modules in `ignored`
    are wrapped in cms.ignore.
    """

    if not module_names:
        raise ValueError("Offload path must contain at least one module")

    modules = []
    for name in module_names:
        if not hasattr(process, name):
            raise AttributeError(f"Process has no module named '{name}'")
        module = getattr(process, name)
        modules.append(cms.ignore(module) if name in ignored else module)

    # Chain modules with +
    sequence = modules[0]
    for mod in modules[1:]:
        sequence = sequence + mod

    path = cms.Path(sequence)
    add_new(process, path_name, path)

    if hasattr(process, "schedule") and process.schedule is not None:
        process.schedule.append(path)

    return path

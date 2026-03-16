# This module contains functions to edit local and remote processes
from typing import Dict, List

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *
from HeterogeneousCore.MPICore.modules import *

def add_controller_to_local(process):
    process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    process.MPIService.pmix_server_uri = "file:server.uri"
    process.mpiController = MPIController()
    # Multiple luminocity blocks are currently unsupported
    process.options.numberOfConcurrentLuminosityBlocks = 1


def clone_module_from_process(dst, src, name):
    """
    Clone module 'name' from src Process into dst Process.
    """
    if not hasattr(src, name):
        raise RuntimeError(f"Module {name} not found in source process")

    setattr(dst, name, getattr(src, name).clone())


def create_remote_process(local_process, modules_to_run):
    remote_process = cms.Process("REMOTE")
    remote_process.load("Configuration.StandardSequences.Accelerators_cff")

    # load the global psets and event setup modules
    for module in local_process.psets.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())
    for module in local_process.es_sources.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())
    for module in local_process.es_producers.keys():
        setattr(remote_process, module, getattr(local_process, module).clone())

    remote_process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    remote_process.MPIService.pmix_server_uri = "file:server.uri"

    # where do i get this firstRun parameter from?
    remote_process.source = cms.Source("MPISource")
    #     firstRun = cms.untracked.uint32(process.Source)
    # )
    remote_process.maxEvents.input = -1

    for module in modules_to_run:
        clone_module_from_process(remote_process, local_process, module)
    
    return remote_process


# -- functions to add senders and receivers --

def is_device_product(prod):
    return prod["type"].startswith("edm::DeviceProduct")

def make_sender_patterns(module_name, products):
    patterns = []

    for p in products:
        if is_device_product(p):
            continue

        friendly_type_name = p["friendly_type_name"]
        label = p["product_instance"]

        patterns.append(f"{friendly_type_name}_{module_name}_{label}_*")

    return patterns


def make_receiver_psets(products):
    psets = []

    for p in products:
        if is_device_product(p):
            continue

        psets.append(
            cms.PSet(
                type=cms.string(p["type"]),
                label=cms.string(p['product_instance'])
            )
        )

    return cms.VPSet(*psets)


def make_grouped_receiver_psets(products):
    psets = []

    for p in products:
        if is_device_product(p):
            continue
        
        if p["product_instance"] == "":
            label = p["module"]
        else:
            label = f"{p['module']}@{p['product_instance']}"

        psets.append(
            cms.PSet(
                type=cms.string(p["type"]),
                label=cms.string(label)
            )
        )

    return cms.VPSet(*psets)


def replace_module(process, name, new_module):
    if hasattr(process, name):
        delattr(process, name)
    setattr(process, name, new_module)


def create_sender(
    module_name,
    products,
    instance,
    sender_upstream,
    path_state_capture=None,
):
    """
    Add MPISender for one module.
    """
    sender_products = make_sender_patterns(module_name, products)

    if path_state_capture is not None:
        sender_products.append(f"*_{path_state_capture}__*".replace(" ", ""))

    sender = cms.EDProducer(
        "MPISender",
        upstream=cms.InputTag(sender_upstream),
        instance=cms.int32(instance),
        products=cms.vstring(*sender_products),
    )

    return sender


def create_group_sender(
    group,
    all_products,
    instance,
    upstream_module,
    path_state_capture=None,
):
    """
    Add MPISender for multiple modules.
    """
    sender_products = []
    for offloaded_module in group:
        sender_products.extend(make_sender_patterns(offloaded_module, all_products[offloaded_module]))

    if path_state_capture is not None:
        sender_products.append(f"*_{path_state_capture}__*".replace(" ", ""))

    sender = cms.EDProducer(
        "MPISender",
        upstream=cms.InputTag(upstream_module),
        instance=cms.int32(instance),
        products=cms.vstring(*sender_products),
    )

    return sender


def create_group_receiver(
    group,
    all_products,
    instance,
    receiver_upstream,
    path_state_capture=False,
):
    """
    MPIReceiver for one module.
    """
    receiver_products = []
    for offloaded_module in group:
        receiver_products.extend(make_grouped_receiver_psets(all_products[offloaded_module]))

    if path_state_capture:
        receiver_products.append(
            cms.PSet(
                type=cms.string("edm::PathStateToken"),
                label=cms.string(""),
            )
        )

    receiver = cms.EDProducer(
        "MPIReceiver",
        upstream=cms.InputTag(receiver_upstream),
        instance=cms.int32(instance),
        products=cms.VPSet(*receiver_products),
    )

    return receiver


def create_receiver(
    products,
    instance,
    receiver_upstream,
    path_state_capture=False,
):
    """
    MPIReceiver for one module.
    """
    receiver_products = make_receiver_psets(products)

    if path_state_capture:
        receiver_products.append(
            cms.PSet(
                type=cms.string("edm::PathStateToken"),
                label=cms.string(""),
            )
        )

    receiver = cms.EDProducer(
        "MPIReceiver",
        upstream=cms.InputTag(receiver_upstream),
        instance=cms.int32(instance),
        products=cms.VPSet(*receiver_products),
    )

    return receiver


def create_receiver_alias(receiver_name,
    products,
    module_name
):
    """
    Create module aliases for receiver group
    """
    psets = []

    for p in products:
        if is_device_product(p):
            continue
        
        if p["product_instance"] == "":
            fromProductInstance_string = module_name
        else:
            fromProductInstance_string = f"{module_name}@{p['product_instance']}"

        psets.append(
            cms.PSet(
                type=cms.string(p["friendly_type_name"]),
                fromProductInstance=cms.string(fromProductInstance_string),
                toProductInstance = cms.string(p["product_instance"])
            )
        )
    
    alias = cms.EDAlias(
            **{
                receiver_name: cms.VPSet(*psets)
            }
        )

    return alias
    


def make_new_path(
    process,
    path_name: str,
    module_names: list[str],
):
    """
    Create a cms.Path from an ordered list of module names
    and optionally append it to the process schedule.
    """

    if not module_names:
        raise ValueError("Offload path must contain at least one module")

    modules = []
    for name in module_names:
        if not hasattr(process, name):
            raise AttributeError(f"Process has no module named '{name}'")
        modules.append(getattr(process, name))

    # Chain modules with +
    sequence = modules[0]
    for mod in modules[1:]:
        sequence = sequence + mod

    path = cms.Path(sequence)
    setattr(process, path_name, path)

    if hasattr(process, "schedule") and process.schedule is not None:
        process.schedule.append(path)

    return path

# This module contains functions to edit local and remote processes
from typing import Dict, List

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import *
from HeterogeneousCore.MPICore.mpiController_cfi import mpiController as mpiController_


# -- functions add communicaion base ---


def add_controller_to_local(process):
    process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
    process.MPIService.pmix_server_uri = "file:server.uri"
    process.mpiController = mpiController_.clone()


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

    # load the event setup - uncomment later
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

        instance = p["instance"]
        label = p["product_instance"]

        if label:
            patterns.append(f"{instance}_{module_name}_{label}_*")
        else:
            patterns.append(f"{instance}_{module_name}__*")

    return patterns


def make_receiver_psets(products):
    psets = []

    for p in products:
        if is_device_product(p):
            continue

        psets.append(
            cms.PSet(
                type=cms.string(p["type"]),
                label=cms.string(p["product_instance"] or "")
            )
        )

    return cms.VPSet(*psets)



def replace_module(process, name, new_module):
    if hasattr(process, name):
        delattr(process, name)
    setattr(process, name, new_module)




def get_sender_name(module_name):
    return f"mpiSender{module_name[0].upper()}{module_name[1:]}"


def create_sender(
    module_name,
    products,
    instance,
    sender_upstream,
    path_state_capture=None,
):
    """
    Add MPISender (local) for one module.
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





def create_receiver(
    products,
    instance,
    receiver_upstream,
    path_state_capture=False,
):
    """
    MPIReceiver (remote) for one module.
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

def add_activity_filter(process, module_name):
    filter_object =  cms.EDFilter("PathStateRelease",
            state = cms.InputTag(module_name)
        )
    filter_name = f"activityFilter{module_name}"
    setattr(process, filter_name, filter_object)
    return filter_name




def add_sender_receiver(
    sender_process,
    receiver_process,
    module_name,
    products,
    instance,
    sender_upstream,
    receiver_upstream,
    path_state_capture=None,
):
    """
    Add MPISender (local) + MPIReceiver (remote) for one module.
    """

    # ----------------
    # Local: MPISender
    # ----------------
    sender_name = f"mpiSender{module_name[0].upper()}{module_name[1:]}"
    sender_products = make_sender_patterns(module_name, products)

    if path_state_capture is not None:
        sender_products.append(f"*_{path_state_capture}__*".replace(" ", ""))

    sender = cms.EDProducer(
        "MPISender",
        upstream=cms.InputTag(sender_upstream),
        instance=cms.int32(instance),
        products=cms.vstring(*sender_products),
    )

    setattr(sender_process, sender_name, sender)

    # ------------------
    # Remote: MPIReceiver
    # ------------------
    receiver_products = make_receiver_psets(products)

    if path_state_capture is not None:
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

    replace_module(receiver_process, module_name, receiver)

    return sender_name





# -- edit path --

def make_new_path(
    process,
    path_name: str,
    module_names: list[str],
    append_to_schedule: bool = True,
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

    if append_to_schedule:
        if not hasattr(process, "schedule") or process.schedule is None:
            process.schedule = cms.Schedule()
        process.schedule.append(path)

    return path


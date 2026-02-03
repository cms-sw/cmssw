#!/usr/bin/env python3

import argparse
import pathlib
import importlib.util
import os
import sys


from cpp_name_getter import CPPNameGetter
from editor_functions import make_new_path, add_sender_receiver, create_remote_process, add_controller_to_local
from module_dependencies_functions import *
# from path_state_helpers import get_sequences_of_the_modules, group_sequences, insert_path_state_capture_by_group

# some unresolved problems:
# how do we decide which of the offloaded modules we send back and which not? (not all hcal modules have their senders)
# do we really have to insert filters after all local receivers? or maybe only before the first module in the dependency group?
# what to do with SerialSync sequences? should we skip tracking their activity even if they contain our module of interest?
# by which criteria can we bake less sequences contain state capture? for pixels, the modules are contained in 4 sequences, but we only inserted state capture in one before
# how to handle more complicated cases of filters?
# what is hltGetRaw and why do we have it in all sequences of manual splits
# should we do something more elegant than dumping processes into files?


def load_config(path: pathlib.Path):
    """
    Load a CMSSW python config as a module.
    Equivalent to: import hlt
    """

    cfg_dir = os.path.dirname(path)

    # Make helper files next to the cfg visible
    sys.path.insert(0, cfg_dir)

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load config from {path}")

        config = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = config
        spec.loader.exec_module(config)
    finally:
        # clean up to avoid polluting global state
        sys.path.pop(0)

    return config


# The splitting algorithm proceeds as follows:
#   1) Form the connected groups from passed modules to separate offloaded modules into distinct paths
#   2) Figure out the dependencies each module has to determine which data has to be transmitted
#   3) Insert senders to local paths with the needed data (raw and/or others)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=pathlib.Path, help="HLT python config")
    parser.add_argument("modules", nargs="+", help="Modules to offload")
    parser.add_argument(
        "-ol",
        "--output_local",
        type=pathlib.Path,
        default=pathlib.Path("splitted_config/local_split.py"),
        help="Output config path for the local process",
    )
    parser.add_argument(
        "-or",
        "--output_remote",
        type=pathlib.Path,
        default=pathlib.Path("splitted_config/remote_split.py"),
        help="Output config path for the remote process",
    )
    args = parser.parse_args()

    # Import config
    cfg = load_config(args.config)

    # Sanity check
    if not hasattr(cfg, "process"):
        raise RuntimeError("Config does not define `process`")

    local_process = cfg.process

    modules_to_offload = flatten_all_to_module_list(local_process, args.modules)


    # figure out on which modules the offloaded modules depend 
    # we will send all data products of these dependencies from local to remote

    deps = get_module_dependencies(local_process, modules_to_offload)

    modules_to_send_products_from = deps - set(modules_to_offload)

    print("Data to send from local ", modules_to_send_products_from)

    # determine relationships between modules to be offloaded(not used yet)

    graph = build_restricted_dependency_graph(local_process, set(modules_to_offload))
    groups = connected_groups(graph)

    print("Dependency groups: ", groups)

    # print_dependency_groups(graph)


    # base_sequences = get_sequences_of_the_modules(local_process, modules_to_offload)

    # print("Base sequences: ", base_sequences)

    # sequences_by_group = group_sequences(groups, base_sequences)

    # print(sequences_by_group)

    # # insert path state caprtures on local
    
    
    # capture_names = insert_path_state_capture_by_group(local_process, groups, sequences_by_group)


    # -- get c++ names --

    print("Launching cmsRun to get c++ names of all products in the process...")

    cpp_names_getter = CPPNameGetter(local_process, exists=False)
    cpp_names_of_the_products = cpp_names_getter.get_cpp_types_of_module_products()


    # --- start editing ---

    add_controller_to_local(local_process)
    remote_process = create_remote_process(local_process, modules_to_offload)

    mpi_path_modules_local = ["mpiController"]
    mpi_path_modules_remote = []

    instance = 1

    # how to add state captures to the splitter?
    # (uncomplicated implementation)

    # 1) for the local sender - unite the sequences by group and insert group capture in the beginning of each sequence
    # add this activity to the products of sender
    # 2) for the remote receiver - add path state product to the list
    # add the activity filter at the beginning of the group path

    # 3) for the remote sender - add the state capture at the end of the groups path
    # each sender module depens on this state capture
    # 4) for the local receiver - before deleting the offloaded module, insert filter for the activity it receives

    # send the data needed by offloaded modules from local to remote
    for local_dependency in modules_to_send_products_from:
        sender_name = add_sender_receiver(
            sender_process=local_process,
            receiver_process=remote_process,
            module_name=local_dependency,
            products=cpp_names_of_the_products[local_dependency],
            instance=instance,
            sender_upstream="mpiController",
            receiver_upstream="source",
        )
        instance += 1
        mpi_path_modules_local.append(sender_name)
        mpi_path_modules_remote.append(local_dependency)
    

    for group in groups:
        # send the results from remote to local
        for i, offloaded_module in enumerate(group):
            if i == 0:
                sender_name_remote = add_sender_receiver(
                    sender_process=remote_process,
                    receiver_process=local_process,
                    module_name=offloaded_module,
                    products=cpp_names_of_the_products[offloaded_module],
                    instance=instance,
                    sender_upstream=mpi_path_modules_remote[0], # needs to be changed after multiple groups are supported
                    receiver_upstream=mpi_path_modules_local[-1], # and here as well 
                )
            else:
                sender_name_remote = add_sender_receiver(
                    sender_process=remote_process,
                    receiver_process=local_process,
                    module_name=offloaded_module,
                    products=cpp_names_of_the_products[offloaded_module],
                    instance=instance,
                    sender_upstream=mpi_path_modules_remote[-1], # needs to be changed after multiple groups are supported
                    receiver_upstream=mpi_path_modules_local[-1], # and here as well 
                )
            instance += 1
            mpi_path_modules_local.append(offloaded_module)
            mpi_path_modules_remote.append(sender_name_remote)
    

    # add all needed paths to the processed and schedule them

    make_new_path(local_process, "Offload", mpi_path_modules_local)
    make_new_path(remote_process, "MPIPath", mpi_path_modules_remote)
    for i, group in enumerate(groups):
        make_new_path(remote_process, "RemoteOffloadedSequence"+str(i), group)

    # dump the 2 processes into files

    args.output_local.parent.mkdir(parents=True, exist_ok=True)
    args.output_remote.parent.mkdir(parents=True, exist_ok=True)

    args.output_local.write_text(local_process.dumpPython())
    args.output_remote.write_text(remote_process.dumpPython())

    print("Success!")



if __name__ == "__main__":
    main()

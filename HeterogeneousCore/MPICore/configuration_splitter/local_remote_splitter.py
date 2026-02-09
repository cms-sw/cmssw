#!/usr/bin/env python3

import argparse
import pathlib
import importlib.util
import os
import sys


from cpp_name_getter import CPPNameGetter
from editor_functions import *
from module_dependencies_functions import *
from path_state_helpers import *

# some unresolved problems:
# how do we decide which of the offloaded modules we send back and which not? (not all hcal modules have their senders) - check by input tags and add info about it, document, etc. - done? (maybe use DependencyGraph later)
# do we really have to insert filters after all local receivers? or maybe only before the first module in the dependency group? - put just one before the receiver of the first offloaded module of the dependency group
# should we do something more elegant than dumping processes into files? - no, fine for now


# filters - for when local offloads several


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
    parser.add_argument("modules", nargs="+", help="Modules and/or sequences to offload")
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
    parser.add_argument(
        "--shared-modules",
        nargs="+",
        default=[],
        help="Modules that must run on both local and remote processes",
    )
    args = parser.parse_args()

    # Import config
    cfg = load_config(args.config)

    # Sanity check
    if not hasattr(cfg, "process"):
        raise RuntimeError("Config does not define `process`")

    local_process = cfg.process

    pure_modules_to_offload = flatten_all_to_module_list(local_process, args.modules)
    modules_to_run_on_both = flatten_all_to_module_list(local_process, args.shared_modules)


    overlap = set(pure_modules_to_offload) & set(modules_to_run_on_both)
    if overlap:
        parser.error(
            f"Modules cannot be both offloaded and shared: {', '.join(overlap)}"
        )
    
    modules_to_offload = pure_modules_to_offload + modules_to_run_on_both


    # figure out on which modules the offloaded modules depend 
    # we will send all data products of these dependencies from local to remote

    deps = get_module_dependencies(local_process, modules_to_offload)

    modules_to_send_products_from = deps - set(modules_to_offload)

    print("Data to send from local ", modules_to_send_products_from)

    # determine relationships between modules to be offloaded(not used yet)

    graph = build_restricted_dependency_graph(local_process, set(modules_to_offload))
    groups = connected_groups(graph)
    print("Dependency groups: ", groups)

    grouped_deps = get_grouped_module_dependencies(local_process, groups)
    print("Grouped depencencies:", grouped_deps )


    producer_to_groups = build_producer_to_groups_map(grouped_deps)
    print("Local producers - dependant groups correspondance: ", producer_to_groups)

    # print_dependency_groups(graph)


    # base_sequences = get_sequences_of_the_modules(local_process, modules_to_offload)

    # # print("Base sequences: ", base_sequences)

    # sequences_by_group = group_sequences(groups, base_sequences)

    # print("Sequences by group: ", sequences_by_group)

    # find correspondance modules_to_send_products_from - sequences (on which sequences do we need the data products of this module?)
   
    
    # capture_names = insert_path_state_capture_by_group(local_process, groups, sequences_by_group)
    


    # -- get c++ names --

    print("Launching cmsRun to get c++ names of all products in the process...")

    cpp_names_getter = CPPNameGetter(local_process, exists=True)
    cpp_names_of_the_products = cpp_names_getter.get_cpp_types_of_module_products()


    # --- start editing ---

    add_controller_to_local(local_process)
    remote_process = create_remote_process(local_process, modules_to_offload)

    mpi_path_modules_local = ["mpiController"]
    mpi_path_modules_remote = []

    instance = 1

    # how to add state captures to the splitter?
    # (uncomplicated implementation)

    # 1) for the local sender - create path state capture per sender. 
    # insert this path state capture into before the first module of each group (if this group needs it ofc)
    # 
    # 2) for the remote receiver - add path state product to the list
    # add the activity filter at the beginning of the group path

    # 3) for the remote sender - add the state capture at the end of the groups path
    # each sender module depens on this state capture
    # 4) for the local receiver - before deleting the offloaded module, insert filter for the activity it receives


    # send the data needed by offloaded modules from local to remote
    remote_filters_by_group = [[] for _ in range(len(groups))]
    local_sender_by_group = [[] for _ in range(len(groups))]
    for local_dependency, group_indices in producer_to_groups.items():
        first_dependency_in_a_group = [groups[i][0] for i in group_indices]
        capture_name = insert_path_state_capture_before(local_process, first_modules_in_a_group=first_dependency_in_a_group, capture_name=local_dependency)
        sender = create_sender(
                module_name=local_dependency,
                products=cpp_names_of_the_products[local_dependency],
                instance=instance,
                sender_upstream="mpiController",
                path_state_capture = capture_name
            )
        sender_name = get_sender_name(local_dependency)
        setattr(local_process, sender_name, sender)



        receiver = create_receiver(
                products=cpp_names_of_the_products[local_dependency],
                instance=instance,
                receiver_upstream="source",
                path_state_capture=True,
            )
        # create filter for the path state
        filter_name = add_activity_filter(remote_process, local_dependency)
        for group_idx in group_indices:
            remote_filters_by_group[group_idx].append(filter_name)
            local_sender_by_group[group_idx].append(sender_name)
        
        setattr(remote_process, local_dependency, receiver)

        instance += 1
        mpi_path_modules_local.append(sender_name)
        mpi_path_modules_remote.append(local_dependency)



        # handle the case when onle local product is needed by multiple remote groups
        # For sending from local process - if data is needed by multiple paths, insert additional path state capture and additional sender per group
        # on remote insert one more receiver for this capture and one more filter in the beginning of the deps group
        # In this approach if there are 2 senders with intersecting groups, it will result in only one group activation sender-receiver pair per group
        if len(group_indices) >= 2:
            for group_idx in group_indices:
                capture_name = insert_path_state_capture_before(local_process, first_modules_in_a_group=[groups[group_idx][0]], capture_name=f"Group{group_idx}Activation")
                sender = create_sender(
                        module_name=local_dependency,
                        products=[],
                        instance=instance,
                        sender_upstream="mpiController",
                        path_state_capture = capture_name
                    )
                sender_name = f"Group{group_idx}ActivationSender"
                setattr(local_process, sender_name, sender)


                receiver = create_receiver(
                        products=[],
                        instance=instance,
                        receiver_upstream="source",
                        path_state_capture=True,
                    )
                receiver_name = f"MPIReceiverGroup{group_idx}Activation"
                setattr(remote_process, receiver_name, receiver)

                # create filter for the path state
                filter_name = add_activity_filter(remote_process, receiver_name)
                remote_filters_by_group[group_idx].append(filter_name)
                local_sender_by_group[group_idx].append(sender_name)

                instance += 1
                mpi_path_modules_local.append(sender_name)
                mpi_path_modules_remote.append(receiver_name)

    

    # get products whose data needs to be sent, excluding modules without local dependencies and modules which should run on both processes
    modules_to_send, modules_without_local_deps = modules_to_send_products_from_by_group(local_process, groups, modules_to_run_on_both)
    print("Offloaded modues whose products need to be sent: ", modules_to_send)

    per_group_remote_captures = []
    # send the results from remote to local

    for group_idx, group in enumerate(modules_to_send):

        remote_capture_name = f"PathStateCaptureGroup{group_idx}"
        setattr(remote_process, remote_capture_name, cms.EDProducer("PathStateCapture"))
        per_group_remote_captures.append(remote_capture_name)
        
        for i, offloaded_module in enumerate(group):
            if i == 0:
                sender_upstream=mpi_path_modules_remote[0]
                receiver_upstream = local_sender_by_group[group_idx][-1]
            else:
                sender_upstream=mpi_path_modules_remote[-1]
                receiver_upstream=mpi_path_modules_local[-1]
            
            sender = create_sender(
                module_name=offloaded_module,
                products=cpp_names_of_the_products[offloaded_module],
                instance=instance,
                sender_upstream=sender_upstream,
                path_state_capture = remote_capture_name
            )
            sender_name = get_sender_name(offloaded_module)
            setattr(remote_process, sender_name, sender)

            receiver = create_receiver(
                products=cpp_names_of_the_products[offloaded_module],
                instance=instance,
                receiver_upstream=receiver_upstream,
                path_state_capture=True,
            )

            # insert filter on local before the first module which was supposed to run (is it correct?)
            if i == 0:
                filter_name = add_activity_filter(local_process, offloaded_module)
                insert_modules_before(local_process, getattr(local_process, offloaded_module), getattr(local_process, filter_name))

            delattr(local_process, offloaded_module)
            setattr(local_process, offloaded_module, receiver)


            instance += 1
            mpi_path_modules_local.append(offloaded_module)
            mpi_path_modules_remote.append(sender_name)
    
    # delete offloaded modules whose products are not needed on local from the local process:
    for product in modules_without_local_deps:
        delattr(local_process, product)        
 
    # add all needed paths to the processed and schedule them

    make_new_path(local_process, "Offload", mpi_path_modules_local)
    make_new_path(remote_process, "MPIPath", mpi_path_modules_remote)
    for i, group in enumerate(groups):
        make_new_path(remote_process, "RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+group+[per_group_remote_captures[i]])

    # dump the 2 processes into files

    args.output_local.parent.mkdir(parents=True, exist_ok=True)
    args.output_remote.parent.mkdir(parents=True, exist_ok=True)

    args.output_local.write_text(local_process.dumpPython())
    args.output_remote.write_text(remote_process.dumpPython())

    print("Success!")



if __name__ == "__main__":
    main()

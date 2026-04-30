"""
This module implements the logic to separate one remote process from the local one
"""

import pathlib
import os
import sys

from HeterogeneousCore.MPICore.configuration_splitter.cpp_name_getter import CPPNameGetter
from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import ModuleDependencyAnalyzer, flatten_all_to_module_list
from HeterogeneousCore.MPICore.configuration_splitter.editor_functions import *
from HeterogeneousCore.MPICore.configuration_splitter.path_state_helpers import *
from HeterogeneousCore.MPICore.configuration_splitter.multiple_remotes_option_parser import *

def split_remote(local_process, args, cpp_names_of_the_products):
    modules_to_offload = flatten_all_to_module_list(local_process, args.remote_modules)
    modules_to_run_on_both = flatten_all_to_module_list(local_process, args.duplicate_modules)
    
    # list of all modules to run on remote
    modules_to_offload.extend(m for m in modules_to_run_on_both if m not in modules_to_offload)

    analyzer = ModuleDependencyAnalyzer(local_process)

    groups = analyzer.dependency_groups(modules_to_offload)
    grouped_deps = analyzer.grouped_external_dependencies(groups)
    producer_to_groups = analyzer.producer_to_groups(grouped_deps)

    if args.verbose:
        print("Dependency groups: ", groups)
        print("Grouped depencencies:", grouped_deps )
        print("Local producers - dependant groups correspondance: ", producer_to_groups)

    # get products whose data needs to be sent, excluding modules without local dependencies and modules which should run on both processes
    modules_to_send, modules_without_local_deps = analyzer.modules_to_send_back_by_group(groups, modules_to_run_on_both)

    if args.verbose:
        print("Offloaded modues whose products need to be sent: ", modules_to_send)
        print("Offloaded modules without local dependencies: ", modules_without_local_deps)

    # --- start editing ---

    controller_name = add_controller_to_local(local_process, args.remote_process_name)
    remote_process = create_remote_process(local_process, modules_to_offload, args.remote_process_name, local_process.name_())

    mpi_path_modules_local = [[controller_name] for _ in range(len(groups))]
    mpi_path_modules_remote = [[] for _ in range(len(groups))]

    instance = 1

    # how to add state captures to the splitter?

    # 1) For the local sender - create path state capture per sender. 
    # Insert this path state capture before the first module of each needed group.
    # If multiple groups need these products, create individual path state captures
    # and add separate senders on top
    # 
    # 2) For the remote receiver - add path state product to the list.
    # Add the general activity filter at the beginning of the group path
    # and, if needed, separate activity filters for each group
    #
    # 3) For the remote sender - add the state capture at the end of the groups path.
    # Each sender module for this group depens on this state capture
    #
    # 4) For the local receiver - before deleting the offloaded module, insert filter for the activity it receives

    # send the data needed by offloaded modules from local to remote
    remote_filters_by_group = [[] for _ in range(len(groups))]
    for local_dependency, group_indices in producer_to_groups.items():
        first_dependency_in_a_group = [groups[i][0] for i in group_indices]
        capture_name = f"activityCaptureBefore{local_dependency.title()}"
        insert_path_state_capture_before(local_process, first_modules_in_a_group=first_dependency_in_a_group, capture_name=capture_name)
        sender = create_sender(
                module_name=local_dependency,
                products=cpp_names_of_the_products[local_dependency],
                instance=instance,
                sender_upstream=controller_name,
                path_state_capture = capture_name
            )
        sender_name = f"mpiSender{args.remote_process_name.title()}{local_dependency.title()}"
        setattr(local_process, sender_name, sender)

        receiver = create_receiver(
                products=cpp_names_of_the_products[local_dependency],
                instance=instance,
                receiver_upstream="source",
                path_state_capture=True,
            )
        # create filter for the path state
        filter_name = f"activityFilterAfter{local_dependency.title()}"
        add_activity_filter(remote_process, local_dependency, filter_name)
        setattr(remote_process, local_dependency, receiver)
        for group_idx in group_indices:
            remote_filters_by_group[group_idx].append(filter_name)
            mpi_path_modules_local[group_idx].append(sender_name)
            mpi_path_modules_remote[group_idx].append(local_dependency)

        instance += 1

        # Handle the case when onle local product is needed by multiple remote groups
        # For sending from local process - if data is needed by multiple paths, insert additional path state capture and additional sender per group
        # On remote insert one more receiver for this capture and one more filter in the beginning of the deps group
        # In this approach if there are 2 senders with intersecting groups, it will result in only one group activation sender-receiver pair per group
        if len(group_indices) >= 2:
            for group_idx in group_indices:
                capture_name=f"activityCaptureBefore{args.remote_process_name.title()}Group{group_idx}"
                insert_path_state_capture_before(local_process, first_modules_in_a_group=[groups[group_idx][0]], capture_name=capture_name)
                sender = create_sender(
                        module_name=local_dependency,
                        products=[],
                        instance=instance,
                        sender_upstream=controller_name,
                        path_state_capture = capture_name
                    )
                sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}Activity"
                setattr(local_process, sender_name, sender)

                receiver = create_receiver(
                        products=[],
                        instance=instance,
                        receiver_upstream="source",
                        path_state_capture=True,
                    )
                receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}Activity"
                setattr(remote_process, receiver_name, receiver)

                # create filter for the path state
                filter_name = f"activityFilterBefore{args.remote_process_name.title()}Group{group_idx}"
                add_activity_filter(remote_process, receiver_name, filter_name)
                remote_filters_by_group[group_idx].append(filter_name)

                instance += 1
                mpi_path_modules_local[group_idx].append(sender_name)
                mpi_path_modules_remote[group_idx].append(receiver_name)
    

    per_group_remote_captures = [[] for _ in range(len(groups))]
    
    # send the results from remote to local
    for group_idx, group in enumerate(modules_to_send):
        if len(group)==0:
            continue

        remote_capture_name = f"activityCaptureAfter{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, remote_capture_name, cms.EDProducer("PathStateCapture"))
        per_group_remote_captures[group_idx].append(remote_capture_name)
        
        if len(mpi_path_modules_remote[group_idx]) != 0:
            sender_upstream = mpi_path_modules_remote[group_idx][-1]
        else:
            sender_upstream = "source"
        
        sender = create_group_sender(
            group=group,
            all_products=cpp_names_of_the_products,
            instance=instance,
            upstream_module=sender_upstream,
            path_state_capture=remote_capture_name,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, sender_name, sender)
        
        if len(mpi_path_modules_local[group_idx]) != 0:
            receiver_upstream = mpi_path_modules_local[group_idx][-1]
        else:
            receiver_upstream = controller_name
        
        receiver = create_group_receiver(
            group=group,
            all_products=cpp_names_of_the_products,
            instance=instance,
            receiver_upstream=receiver_upstream,
            path_state_capture=True,
        )
        receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}"
        setattr(local_process, receiver_name, receiver)
        
        instance += 1
        
        # insert filter on local before the first module which was supposed to run (is it correct?)
        filter_name = f"activityFilterAfter{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(local_process, receiver_name, filter_name)
        insert_modules_before(local_process, getattr(local_process, group[0]), getattr(local_process, filter_name))
        
        mpi_path_modules_remote[group_idx].append(sender_name)
        mpi_path_modules_local[group_idx].append(receiver_name)
        
        
        for i, offloaded_module in enumerate(group):
            delattr(local_process, offloaded_module)
            module_alias = create_receiver_alias(receiver_name=receiver_name,
                products=cpp_names_of_the_products[offloaded_module],
                module_name=offloaded_module
            )
            setattr(local_process, offloaded_module, module_alias)

    
    # delete offloaded modules whose products are not needed on local from the local process:
    for product in modules_without_local_deps:
        delattr(local_process, product)        
 
    # add all needed paths to the process and schedule them
    for i, group in enumerate(groups):
        make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}", mpi_path_modules_local[i])
        make_new_path(remote_process, f"MPIPathGroup{i}", mpi_path_modules_remote[i])
        make_new_path(remote_process, args.remote_process_name.title()+"RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+group+per_group_remote_captures[i])
    
    if args.verbose:
        print(f"Successfully split out remote config with name {args.remote_process_name}!")
        
    return remote_process

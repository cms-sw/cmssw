#!/usr/bin/env python

"""
This script splits an HLT configuration into local and remote processes
by offloading selected modules or sequences.

This tool analyzes data dependencies between CMSSW modules and generates
two derived configuration files:
    - a *local* process config
    - a *remote* process config

Modules specified for offloading will be moved to the remote process.
Any required data products are automatically identified and forwarded
between processes unless the producing module is explicitly marked as
shared.

Positional arguments:
    config
        Path to the input HLT Python configuration file.

Required arguments:
    --remote-modules
        Modules to be offloaded to the remote process.
        Sequences can also be passed as a parameter, in which case
        all modules of that sequence will be offloaded

Optional arguments:
    -ol, --output-local
        Path to the output configuration for the local process.
        (default: local.py)

    -or, --output-remote
        Path to the output configuration for the remote process.
        (default: remote.py)

    --duplicate-modules
        List of module labels that must run on both local and remote
        processes. Products from these modules are not transferred
        between processes.
    
    --reuse-cpp-names
        False by default. If this script was run before, pass this 
        argument to reuse the generated file with C++ product names
    
    -v, --verbose
        Print debug outputs

Example 1 (offloading GPU part of ECAL and HCAL):
    python3 local_remote_splitter.py hlt.py --remote-modules hltEcalDigisSoA hltEcalUncalibRecHitSoA \
        hltHcalDigisSoA hltHbheRecoSoA hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \
        --duplicate-modules hltHcalDigis \
        --output-local local.py \
        --output-remote remote.py

Example 2 (offloading GPU part of ECAL, HCAL and pixels):
    python3 local_remote_splitter.py hlt.py --remote-modules hltEcalDigisSoA hltEcalUncalibRecHitSoA \
        hltHcalDigisSoA hltHbheRecoSoA hltParticleFlowRecHitHBHESoA hltParticleFlowClusterHBHESoA \
        hltSiPixelClustersSoA hltSiPixelRecHitsSoA hltPixelTracksSoA hltPixelVerticesSoA \
        --duplicate-modules hltHcalDigis hltOnlineBeamSpot   hltOnlineBeamSpotDevice \
        --output-local local_pixels.py \
        --output-remote remote_pixels.py


Notes:
    - Only data dependencies expressed via InputTag are considered.
    - Module execution order inside dependency groups is preserved.

Some TBDs:
    - Now we check all dependencies by input tags. We have to try utilising DependencyGraph. 
      Modifying the service is mandatory, because current visual graph is to large to be processed
    - (maybe later?) Do more elegant file dumping
    - Test if the results and througputs are the same as in manual split

"""

import argparse
import pathlib
import os
import sys

from HeterogeneousCore.MPICore.configuration_splitter.cpp_name_getter import CPPNameGetter
from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import ModuleDependencyAnalyzer, flatten_all_to_module_set
from HeterogeneousCore.MPICore.configuration_splitter.editor_functions import *
from HeterogeneousCore.MPICore.configuration_splitter.path_state_helpers import *
from FWCore.ParameterSet.processFromFile import processFromFile

def load_config(path: pathlib.Path):
    """
    Load process from the python file
    """

    cfg_dir = os.path.dirname(path)
    # Make helper files next to the cfg visible
    sys.path.insert(0, cfg_dir)

    try:
        process = processFromFile(str(path))
    finally:
        # clean up to avoid polluting global state
        sys.path.pop(0)

    return process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=pathlib.Path, help="HLT python config")
    parser.add_argument(
        "--remote-modules",
        nargs="+",
        default=[],
        help="Modules and/or sequences to offload",
    )
    parser.add_argument(
        "-ol",
        "--output-local",
        type=pathlib.Path,
        default=pathlib.Path("local.py"),
        help="Output config path for the local process",
    )
    parser.add_argument(
        "-or",
        "--output-remote",
        type=pathlib.Path,
        default=pathlib.Path("remote.py"),
        help="Output config path for the remote process",
    )
    parser.add_argument(
        "--duplicate-modules",
        nargs="+",
        default=[],
        help="Modules that must run on both local and remote processes",
    )
    parser.add_argument(
        "--reuse-cpp-names",
        action="store_true",
        help="Assume the file with C++ product names already exists; "
            "do not run cmsRun to regenerate it",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print execution logs of the splitter",
    )

    args = parser.parse_args()

    local_process = load_config(args.config)

    modules_to_offload = flatten_all_to_module_set(local_process, args.remote_modules)
    modules_to_run_on_both = flatten_all_to_module_set(local_process, args.duplicate_modules)
    
    # list of all modules to run on remote
    modules_to_offload |= modules_to_run_on_both

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


    # -- get c++ names --

    if args.verbose:
        if not args.reuse_cpp_names:
            print("Launching cmsRun to get C++ names of all products in the process...")
        else:
            print("Truing to get C++ product names from existing file...")

    cpp_names_getter = CPPNameGetter(local_process, reuse=args.reuse_cpp_names)
    cpp_names_of_the_products = cpp_names_getter.get_cpp_types_of_module_products()

    if args.verbose:
        print("Got C++ product names")
    

    # --- start editing ---

    add_controller_to_local(local_process)
    remote_process = create_remote_process(local_process, modules_to_offload)

    mpi_path_modules_local = ["mpiController"]
    mpi_path_modules_remote = []

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



        # Handle the case when onle local product is needed by multiple remote groups
        # For sending from local process - if data is needed by multiple paths, insert additional path state capture and additional sender per group
        # On remote insert one more receiver for this capture and one more filter in the beginning of the deps group
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

    


    per_group_remote_captures = [[] for _ in range(len(groups))]
    
    # send the results from remote to local
    for group_idx, group in enumerate(modules_to_send):
        if len(group)==0:
            continue

        remote_capture_name = f"PathStateCaptureGroup{group_idx}"
        setattr(remote_process, remote_capture_name, cms.EDProducer("PathStateCapture"))
        per_group_remote_captures[group_idx].append(remote_capture_name)
        
        for i, offloaded_module in enumerate(group):
            if i == 0:
                # First sender in the group has to depend on source 
                sender_upstream=mpi_path_modules_remote[0]
                # First receiver on local depend on the last sender from group or on local controller
                if len(local_sender_by_group[group_idx]) != 0:
                    receiver_upstream = local_sender_by_group[group_idx][-1]
                else:
                    receiver_upstream = "mpiController"
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
        make_new_path(remote_process, "RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+group+per_group_remote_captures[i])

    # dump the 2 processes into files

    args.output_local.parent.mkdir(parents=True, exist_ok=True)
    args.output_remote.parent.mkdir(parents=True, exist_ok=True)

    args.output_local.write_text(local_process.dumpPython())
    args.output_remote.write_text(remote_process.dumpPython())
    print("Success!")



if __name__ == "__main__":
    main()

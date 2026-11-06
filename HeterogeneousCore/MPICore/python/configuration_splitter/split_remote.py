"""
This module implements the logic to separate one remote process from the local one
"""

from collections import defaultdict

import FWCore.ParameterSet.Config as cms
from HLTrigger.Configuration.common import insert_modules_before

from HeterogeneousCore.MPICore.configuration_splitter.module_dependency_analyzer import (
    SOURCE,
    ModuleDependencyAnalyzer,
)
from HeterogeneousCore.MPICore.configuration_splitter.process_builders import (
    add_activity_filter,
    add_controller_to_local,
    alias_entry,
    create_alias,
    create_receiver,
    create_remote_process,
    create_sender,
    insert_path_state_capture_before,
    is_device_product,
    make_new_path,
)


def _flatten_all_to_module_list(process, user_args):
    """
    Turn a provided module list into an ordered list of labels. Unknown modules
    are skipped. Duplicates are dropped.
    """
    module_list = []
    seen = set()

    for name in user_args:
        if not hasattr(process, name):
            print(f"[WARN] process has no attribute named '{name}'")
            continue

        obj_ = getattr(process, name)

        if hasattr(obj_, "moduleNames"):
            for mod in obj_.moduleNames():
                if mod not in seen:
                    module_list.append(mod)
                    seen.add(mod)
        else:
            if name not in seen:
                module_list.append(name)
                seen.add(name)

    return module_list


def _products_to_forward(consumed, process_name):
    """
    turns the list of `ConsumedProduct`s (from the json dependency graph) into the distinct data products to send.
    """
    products = {}
    registered = {}
    for c in consumed:
        product = c.product
        # Device products (edm::DeviceProduct<T>) cannot be sent by the
        # MPISender. Under the same label and product instance, the host copy of
        # these products should have been registered by the FW. Consumers on the
        # other process receive the host copy, so the device product is skipped
        # here. Sending device products (via de MPISenderPortable) will be
        # supported soon.
        if is_device_product(product):
            continue
        if product["process"] != process_name:
            product = dict(product, sender_process=c.tag[2])
        key = (product["module"], product["product_instance"], product["friendly_type_name"], product.get("sender_process", ""))
        if key in products:
            continue

        # a receiver can register a given type under a given product instance only once
        branch = alias_entry(product, "", grouped=True)[:2]
        if branch in registered:
            raise RuntimeError(
                f"the data products {registered[branch]} and {key} would be registered twice by the same MPIReceiver"
            )
        registered[branch] = key
        products[key] = product
    return list(products.values())


def split_remote(local_process, args, cpp_names_of_the_products, dependency_graph):
    """
    Split one remote process out of the local process. Edit `local_process` in
    place, and return the new remote process.
    """
    # Convert the provided module lists to an ordered list of module labels.
    modules_to_offload = _flatten_all_to_module_list(local_process, args.remote_modules)
    modules_to_run_on_both = _flatten_all_to_module_list(local_process, args.duplicate_modules)

    modules_to_offload.extend(m for m in modules_to_run_on_both if m not in modules_to_offload)

    # prepare lookup tables from the two dumps
    analyzer = ModuleDependencyAnalyzer(dependency_graph, cpp_names_of_the_products)

    # the units to offload and to gate together
    groups = analyzer.dependency_groups(modules_to_offload)

    # the local producers the groups consume from, and the data products that have to be sent over
    producer_to_groups, consumed_by_producer = analyzer.external_dependencies_by_group(groups)

    # nothing forwards run, lumi or process block products, so warn about those dependencies
    for module, producers in sorted(analyzer.non_event_dependencies(modules_to_offload).items()):
        print(
            f"[WARN] offloaded module '{module}' consumes a run, lumi or process block product of "
            f"{', '.join(repr(producer) for producer in sorted(producers))}, which stay(s) in the local "
            f"process. Nothing forwards those products, so '{module}' will not see them on the remote "
            "process: offload or duplicate the producer(s), or keep this module local."
        )

    if args.verbose:
        print("Dependency groups: ", groups)
        print("Local producers - dependant groups correspondance: ", producer_to_groups)

    # get products whose data needs to be sent, excluding modules without local dependencies and modules which should run on both processes
    modules_to_send, modules_without_local_deps, consumed_back = analyzer.modules_to_send_back_by_group(
        groups, modules_to_run_on_both
    )

    if args.verbose:
        print("Offloaded modues whose products need to be sent: ", modules_to_send)
        print("Offloaded modules without local dependencies: ", modules_without_local_deps)

    # --- start editing ---

    controller_name = add_controller_to_local(local_process, args.remote_process_name)
    remote_process = create_remote_process(local_process, modules_to_offload, args.remote_process_name, local_process.name_())

    # Prepare two lists per group: one with the modules of the future local MPI
    # path, starting with the controller, and another one with the modules of
    # the future remote MPI path.
    mpi_path_modules_local = [[controller_name] for _ in range(len(groups))]
    mpi_path_modules_remote = [[] for _ in range(len(groups))]

    instance = 1

    # the EDAliases the remote process needs so that the offloaded modules find
    # each data product under the label they ask for
    remote_aliases = defaultdict(lambda: defaultdict(set))

    # send the data needed by offloaded modules from local to remote
    remote_filters_by_group = [[] for _ in range(len(groups))]
    for local_dependency, group_indices in producer_to_groups.items():
        # every member of every group that needs this dependency
        consumers_in_groups = [module_name for i in group_indices for module_name in groups[i]]
        # only the data products the offloaded modules read
        consumed = [c for c in consumed_by_producer[local_dependency] if not is_device_product(c.product)]
        products_for_dependency = _products_to_forward(consumed, analyzer.process_name)

        # the PathStateCapture whose token says the local paths reached those consumers
        capture_name = f"activityCaptureBefore{args.remote_process_name.title()}{local_dependency.title()}"
        # insert it in front of every one of them, always ahead of the filter that waits for it
        insert_path_state_capture_before(local_process, first_modules_in_a_group=consumers_in_groups, capture_name=capture_name)
        # send the products, or, when the token is missing, the news that the path was not reached
        sender = create_sender(
                products=products_for_dependency,
                instance=instance,
                upstream=controller_name,
                activity=capture_name
            )
        sender_name = f"mpiSender{args.remote_process_name.title()}{local_dependency.title()}"
        setattr(local_process, sender_name, sender)

        if local_dependency == SOURCE:
            # "source" is a reserved cms.Source slot, so the receiver takes a label of its own and
            # carries the data products of the Source and of the input together
            receiver = create_receiver(
                    products=products_for_dependency,
                    instance=instance,
                    upstream="source",
                    activity=True,
                    grouped=True,
                )
            receiver_name = f"mpiReceiver{args.remote_process_name.title()}{local_dependency.title()}"
            setattr(remote_process, receiver_name, receiver)

            # every label the offloaded modules ask for becomes an EDAlias of the receiver
            for c in consumed:
                remote_aliases[c.tag[0]][receiver_name].add(alias_entry(c.product, c.instance, grouped=True))
        else:
            # an ordinary producer: the receiver takes its module label over
            receiver = create_receiver(
                    products=products_for_dependency,
                    instance=instance,
                    upstream="source",
                    activity=True,
                )
            receiver_name = local_dependency
            setattr(remote_process, receiver_name, receiver)

            # the labels of the EDAliases of the producer become EDAliases of the receiver
            for c in consumed:
                if c.tag[0] != local_dependency:
                    remote_aliases[c.tag[0]][receiver_name].add(alias_entry(c.product, c.instance, grouped=False))

        # create filter for the path state
        filter_name = f"activityFilterAfter{local_dependency.title()}"
        add_activity_filter(remote_process, receiver_name, filter_name)
        for group_idx in group_indices:
            remote_filters_by_group[group_idx].append(filter_name)
            mpi_path_modules_local[group_idx].append(sender_name)
            mpi_path_modules_remote[group_idx].append(receiver_name)

        instance += 1


    # recreate on the remote the EDAliases through which offloaded modules read other offloaded modules
    offloaded = {m for group in groups for m in group}
    for module in offloaded:
        for c in analyzer.consumed_products(module):
            if c.origin in offloaded and c.tag[0] != c.origin:
                remote_aliases[c.tag[0]][c.origin].add(alias_entry(c.product, c.instance, grouped=False))

    for label, entries in sorted(remote_aliases.items()):
        # a label must be free on the remote; only "source" can be shared, with the MPISource
        if label != "source" and hasattr(remote_process, label):
            raise RuntimeError(f"the remote process already has a '{label}', so it cannot be an EDAlias too")
        setattr(remote_process, label, create_alias(entries))

    # An Activity round trip per group
    for group_idx, group in enumerate(groups):
        # the local PathStateCapture for this group
        capture_name = f"activityCaptureBefore{args.remote_process_name.title()}Group{group_idx}"
        insert_path_state_capture_before(local_process, first_modules_in_a_group=group, capture_name=capture_name)
        # then, a sender carrying only the token
        sender = create_sender(
                products=[],
                instance=instance,
                upstream=controller_name,
                activity=capture_name
            )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}Activity"
        setattr(local_process, sender_name, sender)

        # The corresponding receiver
        receiver = create_receiver(
                products=[],
                instance=instance,
                upstream="source",
                activity=True,
            )
        receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}Activity"
        setattr(remote_process, receiver_name, receiver)

        # and the remote filter, that will pass or not depending on the presence
        # of the token
        filter_name = f"activityFilterBefore{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(remote_process, receiver_name, filter_name)
        remote_filters_by_group[group_idx].append(filter_name)

        mpi_path_modules_local[group_idx].append(sender_name)
        mpi_path_modules_remote[group_idx].append(receiver_name)

        instance += 1

    # the remote PathStateCapture that reports back that a group actually ran
    per_group_remote_captures = [[] for _ in range(len(groups))]
    # the local receiver collecting each group's results, for the groups that send any
    group_receiver_local = [None] * len(groups)
    # the local EDAliases of the offloaded modules
    local_aliases = defaultdict(lambda: defaultdict(set))

    # send the results from remote to local
    for group_idx, group in enumerate(modules_to_send):
        if len(group) == 0:
            continue

        # The PathStateCapture that will tell whether a remote group ran or not
        remote_capture_name = f"activityCaptureAfter{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, remote_capture_name, cms.EDProducer("PathStateCapture"))
        per_group_remote_captures[group_idx].append(remote_capture_name)

        if len(mpi_path_modules_remote[group_idx]) != 0:
            sender_upstream = mpi_path_modules_remote[group_idx][-1]
        else:
            sender_upstream = "source"

        # the data products of this group's members *that the local modules
        # read*.
        group_products = _products_to_forward(consumed_back[group_idx], analyzer.process_name)

        # the remote sender
        sender = create_sender(
            products=group_products,
            instance=instance,
            upstream=sender_upstream,
            activity=remote_capture_name,
        )
        sender_name = f"mpiSender{args.remote_process_name.title()}Group{group_idx}"
        setattr(remote_process, sender_name, sender)

        receiver_upstream = mpi_path_modules_local[group_idx][-1]

        # the local receiver
        receiver = create_receiver(
            products=group_products,
            instance=instance,
            upstream=receiver_upstream,
            activity=True,
            grouped=True,
        )
        receiver_name = f"mpiReceiver{args.remote_process_name.title()}Group{group_idx}"
        setattr(local_process, receiver_name, receiver)

        instance += 1

        # An activity filter so that downstream modules in the local path wait
        # for the results to be received
        filter_name = f"activityFilterAfter{args.remote_process_name.title()}Group{group_idx}"
        add_activity_filter(local_process, receiver_name, filter_name)
        for offloaded_module in group:
            insert_modules_before(local_process, getattr(local_process, offloaded_module), getattr(local_process, filter_name))

        # the sender closes this group's MPI path on the remote
        mpi_path_modules_remote[group_idx].append(sender_name)
        # the receiver goes on a path of its own on local, see below
        group_receiver_local[group_idx] = receiver_name

        for offloaded_module in group:
            # Each offloaded module is deleted from the local process, and
            # replaced by an EDAlias with the same label. The EDAlias points at
            # the receiver.
            delattr(local_process, offloaded_module)
            module_alias = create_alias(
                {
                    receiver_name: {
                        alias_entry(p, p["product_instance"], grouped=True)
                        for p in group_products
                        if p["module"] == offloaded_module
                    }
                }
            )
            setattr(local_process, offloaded_module, module_alias)

        # A local module can read a member of the group through an EDAlias of the member, under a label
        # other than the label of the member (c.tag[0] != c.origin). The data product now comes from the
        # receiver, under the instance "member@instance". Record the entry that the EDAlias needs to
        # point at the receiver. The EDAlias itself is rewritten after the loop over the groups.
        for c in consumed_back[group_idx]:
            if c.tag[0] != c.origin and not is_device_product(c.product):
                local_aliases[c.tag[0]][receiver_name].add(alias_entry(c.product, c.instance, grouped=True))

    # delete from the local process the offloaded modules whose products are not
    # needed on local.
    for product in modules_without_local_deps:
        delattr(local_process, product)

    # Rewrite the local EDAliases that point at a module that now runs only remotely.
    #
    # The loop over the groups above replaced each member with a new EDAlias under the label of the
    # member. The loop below handles the EDAliases that already existed in the configuration, under
    # labels of their own. Such an EDAlias names the member as the module it points to, and the member
    # no longer exists as a module locally.
    #
    # For each such EDAlias:
    #   - drop the entries for remote-only modules;
    #   - keep the entries for other modules;
    #   - add the entries collected in local_aliases, which point at the receivers and cover only the
    #     data products local modules read;
    #   - delete the EDAlias if no entry is left.
    # The new entries point at the receivers directly, not at the new EDAlias of the member, because an
    # EDAlias of an EDAlias is not verified to work.
    remote_only = offloaded - set(modules_to_run_on_both)
    for alias_label, alias in list(local_process.aliases_().items()):
        names = alias.parameterNames_()
        if not remote_only.intersection(names):
            continue
        kept = {name: getattr(alias, name) for name in names if name not in remote_only}
        delattr(local_process, alias_label)
        if kept or alias_label in local_aliases:
            new_alias = create_alias(local_aliases.get(alias_label, {}))
            for name, entries in kept.items():
                setattr(new_alias, name, entries)
            setattr(local_process, alias_label, new_alias)

    # add all needed paths to the process and schedule them
    for i, group in enumerate(groups):
        # Local: Controller, input senders, activity senders
        if mpi_path_modules_local[i]:
            make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}", mpi_path_modules_local[i])
        # Local: Receivers that get the results back
        if group_receiver_local[i] is not None:
            make_new_path(local_process, f"Offload{args.remote_process_name.title()}Group{i}Receive", [group_receiver_local[i]])
        # Remote: Input receivers, activity receivers, results senders
        if mpi_path_modules_remote[i]:
            make_new_path(remote_process, f"MPIPathGroup{i}", mpi_path_modules_remote[i])
        # Remote: the path that runs the members of the group. The path contains, in order:
        #   - the activity filters of the group: one per local producer of the inputs of the group
        #     (activityFilterAfter<Origin>), which pass if the inputs arrived, and one for the group
        #     (activityFilterBefore<Remote>Group<i>), which passes if the local paths reached the group;
        #   - the members of the group, in dependency order;
        #   - the remote capture (activityCaptureAfter<Remote>Group<i>), for the groups that send results
        #     back. The capture tells the local process that the group ran.
        # Together, the filters make the members run on the remote process for the same events as in the
        # original configuration.
        make_new_path(remote_process, args.remote_process_name.title()+"RemoteOffloadedSequence"+str(i), remote_filters_by_group[i]+group+per_group_remote_captures[i])

    # instance 0 is reserved for the MPIController/MPISource pair, and the MPI tag caps at 255
    if instance > 256:
        raise RuntimeError(
            f"splitting out '{args.remote_process_name}' needs {instance - 1} MPI channel instances, "
            "but only 255 are available; offload fewer modules, or split them over several remotes"
        )

    if args.verbose:
        print(f"Successfully split out remote config with name {args.remote_process_name}!")

    return remote_process

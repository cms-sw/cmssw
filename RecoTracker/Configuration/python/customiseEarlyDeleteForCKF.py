import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForCKF(process, products):

    references = collections.defaultdict(list)
    
    if "trackExtenderWithMTD" not in process.producerNames():
        return (products, references)

    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    trajectoryLabels = []
    trackListMergers = []
    def _addProduct(name):
        products[name].append(_branchName("Trajectorys", name))
        products[name].append(_branchName("TrajectorysToOnerecoTracksAssociation", name))
        references[_branchName("TrajectorysToOnerecoTracksAssociation", name)] = [_branchName("Trajectorys", name)]
        trajectoryLabels.append(name)

    for name, module in process.producers_().items():
        cppType = module.type_()
        if cppType == "TrackProducer":
            if module.TrajectoryInEvent:
                _addProduct(name)
        elif cppType == "DuplicateListMerger":
            if module.copyTrajectories:
                _addProduct(name)
        elif cppType == "TrackListMerger":
            trackListMergers.append(module)

    # TrackListMerger copies Trajectory collections silently, so we
    # add its Trajectory products only if we know from above the input
    # has Trajectory collections. Note that this property is transitive.
    def _containsTrajectory(vinputtag):
        for t in vinputtag:
            t2 = t
            if not isinstance(t, cms.VInputTag):
                t2 = cms.InputTag(t)
            for label in trajectoryLabels:
                if t2.getModuleLabel() == label:
                    return True
        return False

    changed = True
    while changed:
        changed = False
        noTrajectoryYet = []
        for tlm in trackListMergers:
            if _containsTrajectory(tlm.TrackProducers):
                _addProduct(tlm.label())
                changed = True
            else:
                noTrajectoryYet.append(tlm)
        trackListMergers = noTrajectoryYet

    return (products, references)

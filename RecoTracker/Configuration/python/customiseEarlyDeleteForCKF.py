import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForCKF(process, products):

    if "trackExtenderWithMTD" not in process.producerNames():
        return products

    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for name, module in process.producers_().items():
        cppType = module._TypedParameterizable__type
        if cppType == "TrackProducer":
            if module.TrajectoryInEvent:
                products[name].append(_branchName("Trajectorys", name))
                products[name].append(_branchName("TrajectorysToOnerecoTracksAssociation", name))
        elif cppType == "DuplicateListMerger":
            if module.copyTrajectories:
                products[name].append(_branchName("Trajectorys", name))
                products[name].append(_branchName("TrajectorysToOnerecoTracksAssociation", name))

    return products

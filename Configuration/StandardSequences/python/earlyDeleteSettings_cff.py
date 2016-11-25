# Abstract all early deletion settings here

import collections

import FWCore.ParameterSet.Config as cms

from RecoTracker.Configuration.customiseEarlyDeleteForSeeding import customiseEarlyDeleteForSeeding

def _hasInputTagModuleLabel(pset, moduleLabel):
    for name in pset.parameterNames_():
        value = getattr(pset,name)
        if isinstance(value, cms.PSet):
            if _hasInputTagModuleLabel(value, moduleLabel):
                return True
        elif isinstance(value, cms.VPSet):
            for ps in value:
                if _hasInputTagModuleLabel(ps, moduleLabel):
                    return True
        elif isinstance(value, cms.VInputTag):
            for t in value:
                t2 = t
                if not isinstance(t, cms.InputTag):
                    t2 = cms.InputTag(t2)
                if t2.getModuleLabel() == moduleLabel:
                    return True
        elif isinstance(value, cms.InputTag):
            if value.getModuleLabel() == moduleLabel:
                return True
    return False

def customiseEarlyDeleteForRECO(process):
    # Mapping label -> [branches]
    # for the producers whose products are to be deleted early
    products = collections.defaultdict(list)

    products = customiseEarlyDeleteForSeeding(process, products)

    # Set process.options.canDeleteEarly
    if not hasattr(process.options, "canDeleteEarly"):
        process.options.canDeleteEarly = cms.untracked.vstring()

    branchSet = set()
    for branches in products.itervalues():
        for branch in branches:
            branchSet.add(branch)
    process.options.canDeleteEarly.extend(list(branchSet))

    # Find the consumers
    for moduleType in [process.producers_(), process.filters_(), process.analyzers_()]:
        for name, module in moduleType.iteritems():
            for producer, branches in products.iteritems():
                if _hasInputTagModuleLabel(module, producer):
                    #print "Module %s mightGet %s" % (name, str(branches))
                    if hasattr(module, "mightGet"):
                        module.mightGet.extend(branches)
                    else:
                        module.mightGet = cms.untracked.vstring(branches)
    return process

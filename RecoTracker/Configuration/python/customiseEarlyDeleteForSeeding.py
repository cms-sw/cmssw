import FWCore.ParameterSet.Config as cms

import collections

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


def customiseEarlyDeleteForSeeding(process):
    # Find the producers
    products = collections.defaultdict(list)
    depends = collections.defaultdict(list)

    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for name, module in process.producers_().iteritems():
        cppType = module._TypedParameterizable__type
        if cppType == "HitPairEDProducer":
            if module.produceSeedingHitSets:
                products[name].append(_branchName("RegionsSeedingHitSets", name))
            if module.produceIntermediateHitDoublets:
                products[name].append(_branchName("IntermediateHitDoublets", name))
        elif cppType in ["PixelTripletHLTEDProducer", "PixelTripletLargeTipEDProducer"]:
            if module.produceSeedingHitSets:
                products[name].append(_branchName("RegionsSeedingHitSets", name))
            if module.produceIntermediateHitTriplets:
                products[name].append(_branchName("IntermediateHitTriplets", name))
            depends[name].append(module.doublets.getModuleLabel())
        elif cppType in ["MultiHitFromChi2EDProducer"]:
            products[name].extend([
                _branchName("RegionsSeedingHitSets", name),
                _branchName("BaseTrackerRecHitsOwned", name)
            ])
        elif cppType == "PixelQuadrupletEDProducer":
            products[name].append(_branchName("RegionsSeedingHitSets", name))
        elif cppType == "PixelQuadrupletMergerEDProducer":
            products[name].extend([
                    _branchName("RegionsSeedingHitSets", name),
                    _branchName("TrajectorySeeds", name)
                    ])

    if len(products) == 0:
        return process

    # Set process.options
    if not hasattr(process, "options"):
        process.options = cms.untracked.PSet()
    if not hasattr(process.options, "canDeleteEarly"):
        process.options.canDeleteEarly = cms.untracked.vstring()
    for branches in products.itervalues():
        process.options.canDeleteEarly.extend(branches)

    # Resolve data dependencies
    #
    # If a productB depends on productA (e.g. by ref or pointer), then
    # everybody that mightGet's producB, must also mightGet productA
    def _resolve(keys, name):
        for dependsOn in depends[name]:
            if dependsOn in keys:
                _resolve(keys, dependsOn)
                keys.remove(dependsOn)
            products[name].extend(products[dependsOn])

    keys = set(depends.keys())
    while len(keys) > 0:
        name = keys.pop()
        _resolve(keys, name)

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

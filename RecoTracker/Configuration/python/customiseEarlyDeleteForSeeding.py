import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForSeeding(process, products):
    # Find the producers
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
        return products

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

    return products

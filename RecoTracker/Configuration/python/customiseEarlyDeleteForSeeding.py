import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForSeeding(process, products):
    # Find the producers
    references = collections.defaultdict(list)

    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for name, module in process.producers_().items():
        cppType = module._TypedParameterizable__type
        if cppType == "HitPairEDProducer":
            if module.produceSeedingHitSets:
                products[name].append(_branchName("RegionsSeedingHitSets", name))
            if module.produceIntermediateHitDoublets:
                products[name].append(_branchName("IntermediateHitDoublets", name))
        elif cppType in ["PixelTripletHLTEDProducer", "PixelTripletLargeTipEDProducer"]:
            dependencies = []
            if module.produceSeedingHitSets:
                products[name].append(_branchName("RegionsSeedingHitSets", name))
                dependencies.append(_branchName("RegionsSeedingHitSets", name))
            if module.produceIntermediateHitTriplets:
                products[name].append(_branchName("IntermediateHitTriplets", name))
                dependencies.append(_branchName("IntermediateHitTriplets", name))
            # LayerHitMapCache of the doublets is forwarded to both
            # products, hence the dependency
            b = _branchName('IntermediateHitDoublets', module.doublets.getModuleLabel())
            for d in dependencies:
                references[d] = [b]
        elif cppType in ["MultiHitFromChi2EDProducer"]:
            products[name].extend([
                _branchName("RegionsSeedingHitSets", name),
                _branchName("BaseTrackerRecHitsOwned", name)
            ])
            references[_branchName("RegionsSeedingHitSets", name)]=[_branchName("BaseTrackerRecHitsOwned", name)]
        elif cppType in ["CAHitQuadrupletEDProducer", "CAHitTripletEDProducer"]:
            products[name].append(_branchName("RegionsSeedingHitSets", name))

    return (products, references)

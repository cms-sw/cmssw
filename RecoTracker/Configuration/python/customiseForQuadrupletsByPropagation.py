import FWCore.ParameterSet.Config as cms

def customiseForQuadrupletsByPropagation(process):
    for module in process._Process__producers.values():
        if not hasattr(module, "SeedMergerPSet"):
            continue

        # Adjust seeding layers
        seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
        seedingLayersModule = getattr(process, seedingLayersName)
        seedingLayersModule.layerList = process.PixelSeedMergerQuadruplets.layerList.value()

        # Configure seed generator / pixel track producer
        del module.SeedMergerPSet
        triplets = module.OrderedHitsFactoryPSet.clone()
        module.OrderedHitsFactoryPSet = cms.PSet(
            ComponentName = cms.string("CombinedHitQuadrupletGenerator"),
            GeneratorPSet = cms.PSet(
                ComponentName = cms.string("PixelQuadrupletGenerator"),
                extraHitRZtolerance = triplets.GeneratorPSet.extraHitRZtolerance,
                extraHitRPhitolerance = triplets.GeneratorPSet.extraHitRPhitolerance,
                maxChi2 = cms.double(50),
                keepTriplets = cms.bool(True)
            ),
            TripletGeneratorPSet = triplets.GeneratorPSet,
            SeedingLayers = cms.InputTag(seedingLayersName),
        )
        if hasattr(triplets.GeneratorPSet, "SeedComparitorPSet"):
            module.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = triplets.GeneratorPSet.SeedComparitorPSet

        if module.type_() == "PixelTrackProducer":
            module.CleanerPSet.useQuadrupletAlgo = cms.bool(True)

    return process

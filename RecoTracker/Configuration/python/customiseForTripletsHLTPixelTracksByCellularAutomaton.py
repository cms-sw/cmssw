import FWCore.ParameterSet.Config as cms


def producers_by_type(process, *types):
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)


def customiseForTripletsHLTPixelTracksByCellularAutomaton(process):
    for module in producers_by_type(process, "PixelTrackProducer"):
        if not hasattr(module, "OrderedHitsFactoryPSet"):
            continue
	pset = getattr(module, "OrderedHitsFactoryPSet")
        if not hasattr(pset, "ComponentName"):
	    continue
	if not (pset.ComponentName == "StandardHitTripletGenerator"):
	    continue    
        # Adjust seeding layers
        seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
   
        # Configure seed generator / pixel track producer
        triplets = module.OrderedHitsFactoryPSet.clone()
        from RecoPixelVertexing.PixelTriplets.CAHitTripletGenerator_cfi import CAHitTripletGenerator as _CAHitTripletGenerator

        module.OrderedHitsFactoryPSet  = _CAHitTripletGenerator.clone(
            ComponentName = cms.string("CAHitTripletGenerator"),
            extraHitRPhitolerance = triplets.GeneratorPSet.extraHitRPhitolerance,
            maxChi2 = cms.PSet(
                pt1    = cms.double(0.9), pt2    = cms.double(2),
                value1 = cms.double(20), value2 = cms.double(10),
                enabled = cms.bool(True),
            ),
            useBendingCorrection = cms.bool(True),
            SeedingLayers = cms.InputTag(seedingLayersName),
            CAThetaCut = cms.double(0.0015),
            CAPhiCut = cms.double(0.01),
            CAHardPtCut = cms.double(0),
        )

        if hasattr(triplets.GeneratorPSet, "SeedComparitorPSet"):
            module.OrderedHitsFactoryPSet.SeedComparitorPSet = triplets.GeneratorPSet.SeedComparitorPSet
    return process

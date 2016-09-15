import FWCore.ParameterSet.Config as cms

def customiseForTripletsByCellularAutomaton(process):
    for module in process._Process__producers.values():
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
        Triplets = module.OrderedHitsFactoryPSet.clone()
        from RecoPixelVertexing.PixelTriplets.CAHitTripletGenerator_cfi import CAHitTripletGenerator as _CAHitTripletGenerator

        module.OrderedHitsFactoryPSet  = _CAHitTripletGenerator.clone(
            ComponentName = cms.string("CAHitTripletGenerator"),
            extraHitRPhitolerance = Triplets.GeneratorPSet.extraHitRPhitolerance,
            maxChi2 = dict(
                pt1    = 0.8, pt2    = 2,
                value1 = 200, value2 = 100,
                enabled = True,
            ),
            useBendingCorrection = True,
            SeedingLayers = cms.InputTag(seedingLayersName),
            CAThetaCut = cms.double(0.00125),
            CAPhiCut = cms.double(0.1),
            CAHardPtCut = cms.double(0),
        )

        if hasattr(Triplets.GeneratorPSet, "SeedComparitorPSet"):
            pset.SeedComparitorPSet = Triplets.GeneratorPSet.SeedComparitorPSet
    return process

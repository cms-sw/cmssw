import FWCore.ParameterSet.Config as cms

def customiseForQuadrupletsByCellularAutomaton(process):
    for module in process._Process__producers.values():
        if not hasattr(module, "OrderedHitsFactoryPSet"):
            continue
	pset = getattr(module, "OrderedHitsFactoryPSet")
        if not hasattr(pset, "ComponentName"):
	    continue
	if not (pset.ComponentName == "CombinedHitQuadrupletGenerator"):
	    continue    
        # Adjust seeding layers
        seedingLayersName = module.OrderedHitsFactoryPSet.SeedingLayers.getModuleLabel()
   


        # Configure seed generator / pixel track producer
        quadruplets = module.OrderedHitsFactoryPSet.clone()
        from RecoPixelVertexing.PixelTriplets.CAHitQuadrupletGenerator_cfi import CAHitQuadrupletGenerator as _CAHitQuadrupletGenerator

        module.OrderedHitsFactoryPSet  = _CAHitQuadrupletGenerator.clone(
            ComponentName = cms.string("CAHitQuadrupletGenerator"),
            extraHitRPhitolerance = quadruplets.GeneratorPSet.extraHitRPhitolerance,
            maxChi2 = dict(
                pt1    = 0.8, pt2    = 2,
                value1 = 200, value2 = 100,
                enabled = True,
            ),
            useBendingCorrection = True,
            fitFastCircle = True,
            fitFastCircleChi2Cut = True,
            SeedingLayers = cms.InputTag(seedingLayersName),
            CAThetaCut = cms.double(0.00125),
            CAPhiCut = cms.double(0.1),
            CAHardPtCut = cms.double(0),
        )

        if hasattr(quadruplets.GeneratorPSet, "SeedComparitorPSet"):
            module.OrderedHitsFactoryPSet.SeedComparitorPSet = quadruplets.GeneratorPSet.SeedComparitorPSet
    return process

import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.IterativeTracking_cff import * # UGLY CIRCULARITY; only needed for the temporary parameter 'whichTracking'

if(whichTracking=='old'):
    newCombinedSeeds = cms.EDProducer(
        "SeedCombiner",
        seedCollections = cms.VInputTag(
        cms.InputTag('iterativeFirstSeeds','FirstPixelTriplets'),
        cms.InputTag('iterativeFirstSeeds','FirstMixedPairs')
        )
        )
else:
    newCombinedSeeds = cms.EDProducer(
        "SeedCombiner",
        seedCollections = cms.VInputTag(
        cms.InputTag('iterativeInitialSeeds','InitialPixelTriplets'),
        cms.InputTag('iterativeLowPtTripletSeeds','LowPtPixelTriplets'),
        )
        )




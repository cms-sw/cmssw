import FWCore.ParameterSet.Config as cms

newCombinedSeeds = cms.EDProducer(
    "SeedCombiner",
    seedCollections = cms.VInputTag(
    cms.InputTag('iterativeInitialSeeds','InitialPixelTriplets'),
    cms.InputTag('iterativeLowPtTripletSeeds','LowPtPixelTriplets'),
    )
    )




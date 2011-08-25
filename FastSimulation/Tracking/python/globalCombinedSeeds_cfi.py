import FWCore.ParameterSet.Config as cms

newCombinedSeeds = cms.EDProducer(
    "SeedCombiner",
    seedCollections = cms.VInputTag(
#        cms.InputTag('iterativeFirstSeeds','FirstPixelTriplets'),
         cms.InputTag('iterativeInitialSeeds','InitialPixelTriplets'),
#        cms.InputTag('iterativeFirstSeeds','FirstMixedPairs')
         cms.InputTag('iterativeLowPtTripletSeeds','LowPtPixelTriplets'),
    )
)




import FWCore.ParameterSet.Config as cms

newCombinedSeeds = cms.EDProducer(
    "SeedCombiner",
    seedCollections = cms.VInputTag(
        cms.InputTag('iterativeFirstSeeds','FirstPixelTriplets'),
        cms.InputTag('iterativeFirstSeeds','FirstMixedPairs')
    )
)




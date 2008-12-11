import FWCore.ParameterSet.Config as cms

newCombinedSeeds = cms.EDFilter(
    "SeedCombiner",
    TripletCollection = cms.InputTag('iterativeFirstSeeds','FirstPixelTriplets'),
    PairCollection = cms.InputTag('iterativeFirstSeeds','FirstMixedPairs')
)



import FWCore.ParameterSet.Config as cms

newCombinedSeeds = cms.EDProducer(
    "SeedCombiner",
    seedCollections = cms.VInputTag(
        cms.InputTag('iterativeInitialSeeds'),
        #cms.InputTag('iterativeLowPtTripletSeeds'),
        cms.InputTag('iterativePixelPairSeeds'),
        cms.InputTag('iterativeMixedTripletStepSeeds'),
        cms.InputTag('iterativePixelLessSeeds'),
    )
)


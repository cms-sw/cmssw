import FWCore.ParameterSet.Config as cms

newCombinedSeeds = cms.EDProducer(
    "SeedCombiner",
    seedCollections = cms.VInputTag(
        cms.InputTag('initialStepSeeds'),
        #cms.InputTag('lowPtTripletStepSeeds'),
        cms.InputTag('pixelPairStepSeeds'),
        cms.InputTag('mixedTripletStepSeeds'),
        cms.InputTag('pixelLessStepSeeds'),
        )
    )


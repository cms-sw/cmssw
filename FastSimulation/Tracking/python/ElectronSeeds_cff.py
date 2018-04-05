import FWCore.ParameterSet.Config as cms

# TODO: sync with RecoTracker/IterativeTracking/python/ElectronSeeds_cff.py

_newCombinedSeeds = cms.EDProducer(
    "SeedCombiner",
    seedCollections = cms.VInputTag(
        cms.InputTag('initialStepSeeds'),
        #cms.InputTag('lowPtTripletStepSeeds'),
        cms.InputTag('pixelPairStepSeeds'),
        cms.InputTag('mixedTripletStepSeeds'),
        cms.InputTag('pixelLessStepSeeds'),
        )
    )


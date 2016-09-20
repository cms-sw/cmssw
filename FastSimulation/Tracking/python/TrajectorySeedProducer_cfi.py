import FWCore.ParameterSet.Config as cms

import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi

trajectorySeedProducer = cms.EDProducer(
    "TrajectorySeedProducer",
    SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi.SeedFromConsecutiveHitsCreator.clone(
        TTRHBuilder = cms.string("WithoutRefit")),
    recHitCombinations = cms.InputTag("fastMatchedTrackerRecHitCombinations"),
    layerList = cms.vstring(),
    seedFinderSelector = cms.PSet(
        measurementTracker = cms.string("")
        )
    )



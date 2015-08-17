import FWCore.ParameterSet.Config as cms

import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi

trajectorySeedProducer = cms.EDProducer(
    "TrajectorySeedProducer",
    SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi.SeedFromConsecutiveHitsCreator.clone(
        TTRHBuilder = cms.string("WithoutRefit")),
    minLayersCrossed = cms.uint32(0),
    layerList = cms.vstring(),
    recHitCombinationss = cms.InputTag("fastMatchedTrackerRecHitCombinations"),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent")
    )



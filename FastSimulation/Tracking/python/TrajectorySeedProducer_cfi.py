import FWCore.ParameterSet.Config as cms

import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi

trajectorySeedProducer = cms.EDProducer(
    "TrajectorySeedProducer",
    SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi.SeedFromConsecutiveHitsCreator.clone(
        TTRHBuilder = cms.string("WithoutRefit")),
    minLayersCrossed = cms.uint32(0),
    recHitCombinations = cms.InputTag("fastMatchedTrackerRecHitCombinations"),
    layerList = cms.vstring(),
    MeasurementTrackerEvent = cms.InputTag("MeasurementTrackerEvent")
    )



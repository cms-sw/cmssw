import FWCore.ParameterSet.Config as cms

import RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi
from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140

trajectorySeedProducer = cms.EDProducer(
    "TrajectorySeedProducer",
    trackingRegions = cms.InputTag(""),
    SeedCreatorPSet = RecoTracker.TkSeedGenerator.SeedFromConsecutiveHitsCreator_cfi.SeedFromConsecutiveHitsCreator.clone(
        TTRHBuilder = cms.string("WithoutRefit")),
    recHitCombinations = cms.InputTag("fastMatchedTrackerRecHitCombinations"),
    seedFinderSelector = cms.PSet(
        measurementTracker = cms.string(""),
        layerList = cms.vstring(),
        #new parameters for phase1 seeding
        BPix = cms.PSet(
            TTRHBuilder = cms.string(''),
            HitProducer = cms.string(''),
            ),
        FPix = cms.PSet(
            TTRHBuilder = cms.string(''),
            HitProducer = cms.string(''),
            ),
        layerPairs = cms.vuint32()
        )
    )


trackingPhase2PU140.toModify(trajectorySeedProducer, recHitCombinations = cms.InputTag("fastTrackerRecHitCombinations"))

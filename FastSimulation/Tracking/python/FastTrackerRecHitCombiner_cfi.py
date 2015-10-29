import FWCore.ParameterSet.Config as cms

fastTrackerRecHitCombinations = cms.EDProducer(
    "FastTrackerRecHitCombiner",
    simHits = cms.InputTag("famosSimHits","TrackerHits"),
    simHit2RecHitMap = cms.InputTag("siTrackerGaussianSmearingRecHits","simHit2RecHitMap"),
    minNHits = cms.uint32(3)
    )

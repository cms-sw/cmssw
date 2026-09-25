import FWCore.ParameterSet.Config as cms

fastTrackerRecHitCombinations = cms.EDProducer(
    "FastTrackerRecHitCombiner",
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    simHit2RecHitMap = cms.InputTag("fastTrackerRecHits","simHit2RecHitMap"),
    minNHits = cms.uint32(3)
    )

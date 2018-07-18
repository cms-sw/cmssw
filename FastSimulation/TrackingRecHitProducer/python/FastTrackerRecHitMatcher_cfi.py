import FWCore.ParameterSet.Config as cms

fastMatchedTrackerRecHits = cms.EDProducer(
    'FastTrackerRecHitMatcher',
    simHits = cms.InputTag("fastSimProducer","TrackerHits"),
    simHit2RecHitMap = cms.InputTag("fastTrackerRecHits","simHit2RecHitMap")
    )

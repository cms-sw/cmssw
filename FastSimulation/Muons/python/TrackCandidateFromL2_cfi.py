import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltL3TrackCandidateFromL2 = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("hltL3TrajectorySeed"),
    OverlapCleaning = cms.bool(True),
    SeedCleaning = cms.bool(True),
    SplitHits = cms.bool(False),
    SimTracks = cms.InputTag('famosSimHits'),
    EstimatorCut = cms.double(200)
)



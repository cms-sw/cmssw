import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrackCandidateProducer_cfi

hltL3TrackCandidateFromL2 = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    SeedProducer = cms.InputTag("hltL3TrajectorySeed"),
    OverlapCleaning = cms.bool(True),
    SeedCleaning = cms.bool(True),
    SplitHits = cms.bool(False),
    TrackProducers = cms.VInputTag(),
    SimTracks = cms.InputTag('famosSimHits'),
    HitBased = cms.untracked.bool(False),
    EstimatorCut = cms.double(200)
)



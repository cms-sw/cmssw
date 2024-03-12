import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrackCandidateProducer_cfi
electronCkfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone(
    src = cms.InputTag("electronMergedSeeds"),
    MinNumberOfCrossedLayers = 5,
    OverlapCleaning = True
    )
# foo bar baz
# 1Z8na142HF5LJ
# KbveJ9sKV01Ra

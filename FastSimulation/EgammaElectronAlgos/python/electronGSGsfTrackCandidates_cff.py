import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrackCandidateProducer_cfi
# GsfTrackCandidateMaker
electronGSGsfTrackCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
electronGSGsfTrackCandidates.SeedProducer = cms.InputTag("electronGSPixelSeeds")
electronGSGsfTrackCandidates.TrackProducers = []
electronGSGsfTrackCandidates.SeedCleaning = True
electronGSGsfTrackCandidates.MinNumberOfCrossedLayers = 5
#electronGSGsfTrackCandidates.SplitHits = False
electronGSGsfTrackCandidates.OverlapCleaning = True

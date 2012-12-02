import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeFourthTrackCandidatesWithPairs = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeFourthTrackCandidates = cms.Sequence(iterativeFourthTrackCandidatesWithPairs)
iterativeFourthTrackCandidatesWithPairs.SeedProducer = cms.InputTag("iterativeFourthSeeds","FourthPixelLessPairs")
iterativeFourthTrackCandidatesWithPairs.TrackProducers = ['firstfilter', 'secfilter','thfilter']
iterativeFourthTrackCandidatesWithPairs.KeepFittedTracks = False
iterativeFourthTrackCandidatesWithPairs.MinNumberOfCrossedLayers = 5


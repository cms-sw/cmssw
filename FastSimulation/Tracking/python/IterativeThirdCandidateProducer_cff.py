import FWCore.ParameterSet.Config as cms

import FastSimulation.Tracking.TrackCandidateProducer_cfi
iterativeThirdTrackCandidatesWithPairs = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
iterativeThirdTrackCandidates = cms.Sequence(iterativeThirdTrackCandidatesWithPairs)
iterativeThirdTrackCandidatesWithPairs.SeedProducer = cms.InputTag("iterativeThirdSeeds","ThirdMixedPairs")
iterativeThirdTrackCandidatesWithPairs.TrackProducers = ['firstfilter', 'secfilter']
iterativeThirdTrackCandidatesWithPairs.KeepFittedTracks = False
iterativeThirdTrackCandidatesWithPairs.MinNumberOfCrossedLayers = 3


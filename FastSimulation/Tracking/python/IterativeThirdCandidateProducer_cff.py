import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
iterativeThirdTrackCandidatesWithPairs = copy.deepcopy(trackCandidateProducer)
iterativeThirdTrackCandidates = cms.Sequence(iterativeThirdTrackCandidatesWithPairs)
iterativeThirdTrackCandidatesWithPairs.SeedProducer = cms.InputTag("iterativeTrackingSeeds","ThirdMixedPairs")
iterativeThirdTrackCandidatesWithPairs.TrackProducer = 'globalPixelGSWithMaterialTracks'
iterativeThirdTrackCandidatesWithPairs.MinNumberOfCrossedLayers = 3


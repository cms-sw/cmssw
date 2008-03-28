import FWCore.ParameterSet.Config as cms

import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
# GsfTrackCandidateMaker
electronGSGsfTrackCandidates = copy.deepcopy(trackCandidateProducer)
electronGSGsfTrackCandidates.SeedProducer = cms.InputTag("electronGSPixelSeeds")
electronGSGsfTrackCandidates.TrackProducer = cms.InputTag("None","None")
electronGSGsfTrackCandidates.SeedCleaning = True
electronGSGsfTrackCandidates.MinNumberOfCrossedLayers = 5


import FWCore.ParameterSet.Config as cms

from FastSimulation.Tracking.IterativeFifthSeedProducer_cff import *
from FastSimulation.Tracking.IterativeFifthCandidateProducer_cff import *
from FastSimulation.Tracking.IterativeFifthTrackProducer_cff import *
from FastSimulation.Tracking.IterativeFifthTrackMerger_cfi import *
from FastSimulation.Tracking.IterativeFifthTrackFilter_cff import *
iterativeFifthTracking = cms.Sequence(iterativeFifthSeeds
                                      +iterativeFifthTrackCandidatesWithPairs
                                      +iterativeFifthTracks
                                      +iterativeFifthTrackMerging
                                      +iterativeFifthTrackFiltering)


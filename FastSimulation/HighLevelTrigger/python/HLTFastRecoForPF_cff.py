import FWCore.ParameterSet.Config as cms
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
hltPFJetCkfTrackCandidates = cms.Sequence(globalPixelTracking)


import FWCore.ParameterSet.Config as cms

from SimMuon.MCTruth.muonSimClassificationByHits_cff import *
from SimTracker.TrackAssociation.trackPrunedMCMatchTask_cff import *

recosim = cms.Task( muonSimClassificationByHitsTask, trackPrunedMCMatchTask )

import FWCore.ParameterSet.Config as cms

from RecoMTD.TimingIDTools.mvaTrainingNtuple_cfi import *
from RecoVertex.Configuration.RecoVertex_cff import unsortedOfflinePrimaryVertices4D

# higher eta cut for the phase 2 tracker
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(mvaTrainingNtuple, TkFilterParameters = cms.PSet(unsortedOfflinePrimaryVertices4D.TkFilterParameters.clone()))

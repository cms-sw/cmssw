import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.PFBRECO_cff  import *

PF2PATTask = cms.Task(PFBRECOTask)
PF2PAT = cms.Sequence(PF2PATTask)

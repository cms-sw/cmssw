import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.photonPFIsolationDepositsPFBRECO_cff import *
from CommonTools.ParticleFlow.Isolation.photonPFIsolationValuesPFBRECO_cff import *

pfPhotonIsolationPFBRECOTask = cms.Task(
    photonPFIsolationDepositsPFBRECOTask ,
    photonPFIsolationValuesPFBRECOTask
    )
pfPhotonIsolationPFBRECOSequence = cms.Sequence(pfPhotonIsolationPFBRECOTask)

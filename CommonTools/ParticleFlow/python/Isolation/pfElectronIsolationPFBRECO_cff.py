import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.electronPFIsolationDepositsPFBRECO_cff import *
from CommonTools.ParticleFlow.Isolation.electronPFIsolationValuesPFBRECO_cff import *

pfElectronIsolationPFBRECOTask = cms.Task(
    electronPFIsolationDepositsPFBRECOTask ,
    electronPFIsolationValuesPFBRECOTask
    )
pfElectronIsolationPFBRECOSequence = cms.Sequence(pfElectronIsolationPFBRECOTask)

import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.electronPFIsolationDepositsPFBRECO_cff import *
from CommonTools.ParticleFlow.Isolation.electronPFIsolationValuesPFBRECO_cff import *

pfElectronIsolationPFBRECOSequence = cms.Sequence(
    electronPFIsolationDepositsPFBRECOSequence +
    electronPFIsolationValuesPFBRECOSequence
    )


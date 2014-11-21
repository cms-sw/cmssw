import FWCore.ParameterSet.Config as cms

# iso deposits and isolation values, defined by the Muon POG

from CommonTools.ParticleFlow.Isolation.muonPFIsolationDepositsPFBRECO_cff import *
from CommonTools.ParticleFlow.Isolation.muonPFIsolationValuesPFBRECO_cff import *

muonPFIsolationPFBRECOSequence =  cms.Sequence(
    muonPFIsolationDepositsPFBRECOSequence +
    muonPFIsolationValuesPFBRECOSequence
)

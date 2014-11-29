import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.muonPFIsolationDepositsPFBRECO_cff import *

#Now prepare the iso deposits
muPFIsoDepositChargedPAT    = muPFIsoDepositChargedPFBRECO.clone()
muPFIsoDepositChargedAllPAT = muPFIsoDepositChargedAllPFBRECO.clone()
muPFIsoDepositNeutralPAT    = muPFIsoDepositNeutralPFBRECO.clone()
muPFIsoDepositGammaPAT      = muPFIsoDepositGammaPFBRECO.clone()
muPFIsoDepositPUPAT         = muPFIsoDepositPUPFBRECO.clone()

muonPFIsolationDepositsPATSequence = cms.Sequence(
    muPFIsoDepositChargedPAT+
    muPFIsoDepositChargedAllPAT+
    muPFIsoDepositGammaPAT+
    muPFIsoDepositNeutralPAT+
    muPFIsoDepositPUPAT
    )

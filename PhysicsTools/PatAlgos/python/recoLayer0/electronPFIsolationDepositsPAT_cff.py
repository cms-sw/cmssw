import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.electronPFIsolationDepositsPFBRECO_cff import *

#Now prepare the iso deposits
elPFIsoDepositChargedPAT    = elPFIsoDepositChargedPFBRECO.clone()
elPFIsoDepositChargedAllPAT = elPFIsoDepositChargedAllPFBRECO.clone()
elPFIsoDepositNeutralPAT    = elPFIsoDepositNeutralPFBRECO.clone()
elPFIsoDepositPUPAT         = elPFIsoDepositPUPFBRECO.clone()
#elPFIsoDepositGammaPAT      = #elPFIsoDepositGammaPFBRECO.clone()
elPFIsoDepositGammaPAT      = elPFIsoDepositGammaPFBRECO.clone()

electronPFIsolationDepositsPATSequence = cms.Sequence(
    elPFIsoDepositChargedPAT+
    elPFIsoDepositChargedAllPAT+
    elPFIsoDepositGammaPAT+
    elPFIsoDepositNeutralPAT+
    elPFIsoDepositPUPAT
    )

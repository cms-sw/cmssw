import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.photonPFIsolationDepositsPFBRECO_cff import *

#Now prepare the iso deposits
phPFIsoDepositChargedPAT    = phPFIsoDepositChargedPFBRECO.clone()
phPFIsoDepositChargedAllPAT = phPFIsoDepositChargedAllPFBRECO.clone()
phPFIsoDepositNeutralPAT    = phPFIsoDepositNeutralPFBRECO.clone()
#phPFIsoDepositGammaPAT      = phPFIsoDepositGammaPFBRECO.clone()
phPFIsoDepositPUPAT         = phPFIsoDepositPUPFBRECO.clone()
phPFIsoDepositGammaPAT      = phPFIsoDepositGammaPFBRECO.clone()

photonPFIsolationDepositsPATSequence = cms.Sequence(
    phPFIsoDepositChargedPAT+
    phPFIsoDepositChargedAllPAT+
    phPFIsoDepositGammaPAT+
    phPFIsoDepositNeutralPAT+
    phPFIsoDepositPUPAT
    )

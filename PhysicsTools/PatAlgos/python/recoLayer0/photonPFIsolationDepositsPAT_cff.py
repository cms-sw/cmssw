import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.photonPFIsolationDepositsPFBRECO_cff as m

#Now prepare the iso deposits
phPFIsoDepositChargedPAT    = m.phPFIsoDepositChargedPFBRECO.clone()
phPFIsoDepositChargedAllPAT = m.phPFIsoDepositChargedAllPFBRECO.clone()
phPFIsoDepositNeutralPAT    = m.phPFIsoDepositNeutralPFBRECO.clone()
#phPFIsoDepositGammaPAT      = m.phPFIsoDepositGammaPFBRECO.clone()
phPFIsoDepositPUPAT         = m.phPFIsoDepositPUPFBRECO.clone()
phPFIsoDepositGammaPAT      = m.phPFIsoDepositGammaPFBRECO.clone()

photonPFIsolationDepositsPATSequence = cms.Sequence(
    phPFIsoDepositChargedPAT+
    phPFIsoDepositChargedAllPAT+
    phPFIsoDepositGammaPAT+
    phPFIsoDepositNeutralPAT+
    phPFIsoDepositPUPAT
    )

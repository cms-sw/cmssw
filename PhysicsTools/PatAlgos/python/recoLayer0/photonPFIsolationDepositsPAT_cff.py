import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.photonPFIsolationDepositsPFBRECO_cff as _m

#Now prepare the iso deposits
phPFIsoDepositChargedPAT    = _m.phPFIsoDepositChargedPFBRECO.clone()
phPFIsoDepositChargedAllPAT = _m.phPFIsoDepositChargedAllPFBRECO.clone()
phPFIsoDepositNeutralPAT    = _m.phPFIsoDepositNeutralPFBRECO.clone()
#phPFIsoDepositGammaPAT      = _m.phPFIsoDepositGammaPFBRECO.clone()
phPFIsoDepositPUPAT         = _m.phPFIsoDepositPUPFBRECO.clone()
phPFIsoDepositGammaPAT      = _m.phPFIsoDepositGammaPFBRECO.clone()

photonPFIsolationDepositsPATSequence = cms.Sequence(
    phPFIsoDepositChargedPAT+
    phPFIsoDepositChargedAllPAT+
    phPFIsoDepositGammaPAT+
    phPFIsoDepositNeutralPAT+
    phPFIsoDepositPUPAT
    )

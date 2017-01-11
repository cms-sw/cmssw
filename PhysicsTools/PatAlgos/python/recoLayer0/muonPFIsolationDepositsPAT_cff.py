import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.muonPFIsolationDepositsPFBRECO_cff as m

#Now prepare the iso deposits
muPFIsoDepositChargedPAT    = m.muPFIsoDepositChargedPFBRECO.clone()
muPFIsoDepositChargedAllPAT = m.muPFIsoDepositChargedAllPFBRECO.clone()
muPFIsoDepositNeutralPAT    = m.muPFIsoDepositNeutralPFBRECO.clone()
muPFIsoDepositGammaPAT      = m.muPFIsoDepositGammaPFBRECO.clone()
muPFIsoDepositPUPAT         = m.muPFIsoDepositPUPFBRECO.clone()

muonPFIsolationDepositsPATSequence = cms.Sequence(
    muPFIsoDepositChargedPAT+
    muPFIsoDepositChargedAllPAT+
    muPFIsoDepositGammaPAT+
    muPFIsoDepositNeutralPAT+
    muPFIsoDepositPUPAT
    )

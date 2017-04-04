import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.muonPFIsolationDepositsPFBRECO_cff as _m

#Now prepare the iso deposits
muPFIsoDepositChargedPAT    = _m.muPFIsoDepositChargedPFBRECO.clone()
muPFIsoDepositChargedAllPAT = _m.muPFIsoDepositChargedAllPFBRECO.clone()
muPFIsoDepositNeutralPAT    = _m.muPFIsoDepositNeutralPFBRECO.clone()
muPFIsoDepositGammaPAT      = _m.muPFIsoDepositGammaPFBRECO.clone()
muPFIsoDepositPUPAT         = _m.muPFIsoDepositPUPFBRECO.clone()

muonPFIsolationDepositsPATTask = cms.Task(
    muPFIsoDepositChargedPAT,
    muPFIsoDepositChargedAllPAT,
    muPFIsoDepositGammaPAT,
    muPFIsoDepositNeutralPAT,
    muPFIsoDepositPUPAT
    )

muonPFIsolationDepositsPATSequence = cms.Sequence(muonPFIsolationDepositsPATTask)

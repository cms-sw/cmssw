import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.electronPFIsolationDepositsPFBRECO_cff as _m

#Now prepare the iso deposits
elPFIsoDepositChargedPAT    = _m.elPFIsoDepositChargedPFBRECO.clone()
elPFIsoDepositChargedAllPAT = _m.elPFIsoDepositChargedAllPFBRECO.clone()
elPFIsoDepositNeutralPAT    = _m.elPFIsoDepositNeutralPFBRECO.clone()
elPFIsoDepositPUPAT         = _m.elPFIsoDepositPUPFBRECO.clone()
#elPFIsoDepositGammaPAT      = #_m.elPFIsoDepositGammaPFBRECO.clone()
elPFIsoDepositGammaPAT      = _m.elPFIsoDepositGammaPFBRECO.clone()

electronPFIsolationDepositsPATTask = cms.Task(
    elPFIsoDepositChargedPAT,
    elPFIsoDepositChargedAllPAT,
    elPFIsoDepositGammaPAT,
    elPFIsoDepositNeutralPAT,
    elPFIsoDepositPUPAT
    )

electronPFIsolationDepositsPATSequence = cms.Sequence(electronPFIsolationDepositsPATTask)

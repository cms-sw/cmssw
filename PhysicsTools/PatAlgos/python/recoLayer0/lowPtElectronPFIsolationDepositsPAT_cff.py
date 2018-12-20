import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.electronPFIsolationDepositsPFBRECO_cff as _m

#Now prepare the iso deposits
lowPtElPFIsoDepositChargedPAT    = _m.elPFIsoDepositChargedPFBRECO.clone()
lowPtElPFIsoDepositChargedAllPAT = _m.elPFIsoDepositChargedAllPFBRECO.clone()
lowPtElPFIsoDepositNeutralPAT    = _m.elPFIsoDepositNeutralPFBRECO.clone()
lowPtElPFIsoDepositPUPAT         = _m.elPFIsoDepositPUPFBRECO.clone()
#elPFIsoDepositGammaPAT      = #_m.elPFIsoDepositGammaPFBRECO.clone()
lowPtElPFIsoDepositGammaPAT      = _m.elPFIsoDepositGammaPFBRECO.clone()

lowPtElectronPFIsolationDepositsPATTask = cms.Task(
    lowPtElPFIsoDepositChargedPAT,
    lowPtElPFIsoDepositChargedAllPAT,
    lowPtElPFIsoDepositGammaPAT,
    lowPtElPFIsoDepositNeutralPAT,
    lowPtElPFIsoDepositPUPAT
    )

lowPtElectronPFIsolationDepositsPATSequence = cms.Sequence(lowPtElectronPFIsolationDepositsPATTask)

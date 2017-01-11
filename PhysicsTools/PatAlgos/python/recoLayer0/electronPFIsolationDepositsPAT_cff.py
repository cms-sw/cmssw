import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.electronPFIsolationDepositsPFBRECO_cff as m

#Now prepare the iso deposits
elPFIsoDepositChargedPAT    = m.elPFIsoDepositChargedPFBRECO.clone()
elPFIsoDepositChargedAllPAT = m.elPFIsoDepositChargedAllPFBRECO.clone()
elPFIsoDepositNeutralPAT    = m.elPFIsoDepositNeutralPFBRECO.clone()
elPFIsoDepositPUPAT         = m.elPFIsoDepositPUPFBRECO.clone()
#elPFIsoDepositGammaPAT      = #m.elPFIsoDepositGammaPFBRECO.clone()
elPFIsoDepositGammaPAT      = m.elPFIsoDepositGammaPFBRECO.clone()

electronPFIsolationDepositsPATSequence = cms.Sequence(
    elPFIsoDepositChargedPAT+
    elPFIsoDepositChargedAllPAT+
    elPFIsoDepositGammaPAT+
    elPFIsoDepositNeutralPAT+
    elPFIsoDepositPUPAT
    )

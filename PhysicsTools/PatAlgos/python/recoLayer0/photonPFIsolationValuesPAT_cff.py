import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.photonPFIsolationValuesPFBRECO_cff as _m

phPFIsoValueCharged03PFIdPAT = _m.phPFIsoValueCharged03PFIdPFBRECO.clone()
phPFIsoValueCharged03PFIdPAT.deposits[0].src = 'phPFIsoDepositChargedPAT'

phPFIsoValueChargedAll03PFIdPAT = _m.phPFIsoValueChargedAll03PFIdPFBRECO.clone()
phPFIsoValueChargedAll03PFIdPAT.deposits[0].src = 'phPFIsoDepositChargedAllPAT'

phPFIsoValueGamma03PFIdPAT = _m.phPFIsoValueGamma03PFIdPFBRECO.clone()
phPFIsoValueGamma03PFIdPAT.deposits[0].src = 'phPFIsoDepositGammaPAT'

phPFIsoValueNeutral03PFIdPAT = _m.phPFIsoValueNeutral03PFIdPFBRECO.clone()
phPFIsoValueNeutral03PFIdPAT.deposits[0].src = 'phPFIsoDepositNeutralPAT'

phPFIsoValuePU03PFIdPAT = _m.phPFIsoValuePU03PFIdPFBRECO.clone()
phPFIsoValuePU03PFIdPAT.deposits[0].src = 'phPFIsoDepositPUPAT'

phPFIsoValueCharged04PFIdPAT = _m.phPFIsoValueCharged04PFIdPFBRECO.clone()
phPFIsoValueCharged04PFIdPAT.deposits[0]. src = 'phPFIsoDepositChargedPAT'

phPFIsoValueChargedAll04PFIdPAT = _m.phPFIsoValueChargedAll04PFIdPFBRECO.clone()
phPFIsoValueChargedAll04PFIdPAT.deposits[0].src = 'phPFIsoDepositChargedAllPAT'

phPFIsoValueGamma04PFIdPAT = _m.phPFIsoValueGamma04PFIdPFBRECO.clone()
phPFIsoValueGamma04PFIdPAT.deposits[0].src = 'phPFIsoDepositGammaPAT'

phPFIsoValueNeutral04PFIdPAT = _m.phPFIsoValueNeutral04PFIdPFBRECO.clone()
phPFIsoValueNeutral04PFIdPAT.deposits[0].src = 'phPFIsoDepositNeutralPAT'

phPFIsoValuePU04PFIdPAT = _m.phPFIsoValuePU04PFIdPFBRECO.clone()
phPFIsoValuePU04PFIdPAT.deposits[0].src = 'phPFIsoDepositPUPAT'

photonPFIsolationValuesPATTask = cms.Task(
    phPFIsoValueCharged03PFIdPAT,
    phPFIsoValueChargedAll03PFIdPAT,
    phPFIsoValueGamma03PFIdPAT,
    phPFIsoValueNeutral03PFIdPAT,
    phPFIsoValuePU03PFIdPAT,
    ##############################
    phPFIsoValueCharged04PFIdPAT,
    phPFIsoValueChargedAll04PFIdPAT,
    phPFIsoValueGamma04PFIdPAT,
    phPFIsoValueNeutral04PFIdPAT,
    phPFIsoValuePU04PFIdPAT
    )

photonPFIsolationValuesPATSequence = cms.Sequence(photonPFIsolationValuesPATTask)

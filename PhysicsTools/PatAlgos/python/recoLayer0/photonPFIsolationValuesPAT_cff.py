import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.photonPFIsolationValuesPFBRECO_cff import *

phPFIsoValueCharged03PFIdPAT = phPFIsoValueCharged03PFIdPFBRECO.clone()
phPFIsoValueCharged03PFIdPAT.deposits[0].src = 'phPFIsoDepositChargedPAT'

phPFIsoValueChargedAll03PFIdPAT = phPFIsoValueChargedAll03PFIdPFBRECO.clone()
phPFIsoValueChargedAll03PFIdPAT.deposits[0].src = 'phPFIsoDepositChargedAllPAT'

phPFIsoValueGamma03PFIdPAT = phPFIsoValueGamma03PFIdPFBRECO.clone()
phPFIsoValueGamma03PFIdPAT.deposits[0].src = 'phPFIsoDepositGammaPAT'

phPFIsoValueNeutral03PFIdPAT = phPFIsoValueNeutral03PFIdPFBRECO.clone()
phPFIsoValueNeutral03PFIdPAT.deposits[0].src = 'phPFIsoDepositNeutralPAT'

phPFIsoValuePU03PFIdPAT = phPFIsoValuePU03PFIdPFBRECO.clone()
phPFIsoValuePU03PFIdPAT.deposits[0].src = 'phPFIsoDepositPUPAT'

phPFIsoValueCharged04PFIdPAT = phPFIsoValueCharged04PFIdPFBRECO.clone()
phPFIsoValueCharged04PFIdPAT.deposits[0]. src = 'phPFIsoDepositChargedPAT'

phPFIsoValueChargedAll04PFIdPAT = phPFIsoValueChargedAll04PFIdPFBRECO.clone()
phPFIsoValueChargedAll04PFIdPAT.deposits[0].src = 'phPFIsoDepositChargedAllPAT'

phPFIsoValueGamma04PFIdPAT = phPFIsoValueGamma04PFIdPFBRECO.clone()
phPFIsoValueGamma04PFIdPAT.deposits[0].src = 'phPFIsoDepositGammaPAT'

phPFIsoValueNeutral04PFIdPAT = phPFIsoValueNeutral04PFIdPFBRECO.clone()
phPFIsoValueNeutral04PFIdPAT.deposits[0].src = 'phPFIsoDepositNeutralPAT'

phPFIsoValuePU04PFIdPAT = phPFIsoValuePU04PFIdPFBRECO.clone()
phPFIsoValuePU04PFIdPAT.deposits[0].src = 'phPFIsoDepositPUPAT'

photonPFIsolationValuesPATSequence = (
    phPFIsoValueCharged03PFIdPAT+
    phPFIsoValueChargedAll03PFIdPAT+
    phPFIsoValueGamma03PFIdPAT+
    phPFIsoValueNeutral03PFIdPAT+
    phPFIsoValuePU03PFIdPAT+
    ##############################
    phPFIsoValueCharged04PFIdPAT+
    phPFIsoValueChargedAll04PFIdPAT+
    phPFIsoValueGamma04PFIdPAT+
    phPFIsoValueNeutral04PFIdPAT+
    phPFIsoValuePU04PFIdPAT
    )

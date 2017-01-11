import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.photonPFIsolationValues_cff import *

phPFIsoValueCharged03PFIdPFBRECO = phPFIsoValueCharged03PFId.clone()
phPFIsoValueCharged03PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositChargedPFBRECO'

phPFIsoValueChargedAll03PFIdPFBRECO = phPFIsoValueChargedAll03PFId.clone()
phPFIsoValueChargedAll03PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositChargedAllPFBRECO'

phPFIsoValueGamma03PFIdPFBRECO = phPFIsoValueGamma03PFId.clone()
phPFIsoValueGamma03PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositGammaPFBRECO'

phPFIsoValueNeutral03PFIdPFBRECO = phPFIsoValueNeutral03PFId.clone()
phPFIsoValueNeutral03PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositNeutralPFBRECO'

phPFIsoValuePU03PFIdPFBRECO = phPFIsoValuePU03PFId.clone()
phPFIsoValuePU03PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositPUPFBRECO'

phPFIsoValueCharged04PFIdPFBRECO = phPFIsoValueCharged04PFId.clone()
phPFIsoValueCharged04PFIdPFBRECO.deposits[0]. src = 'phPFIsoDepositChargedPFBRECO'

phPFIsoValueChargedAll04PFIdPFBRECO = phPFIsoValueChargedAll04PFId.clone()
phPFIsoValueChargedAll04PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositChargedAllPFBRECO'

phPFIsoValueGamma04PFIdPFBRECO = phPFIsoValueGamma04PFId.clone()
phPFIsoValueGamma04PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositGammaPFBRECO'

phPFIsoValueNeutral04PFIdPFBRECO = phPFIsoValueNeutral04PFId.clone()
phPFIsoValueNeutral04PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositNeutralPFBRECO'

phPFIsoValuePU04PFIdPFBRECO = phPFIsoValuePU04PFId.clone()
phPFIsoValuePU04PFIdPFBRECO.deposits[0].src = 'phPFIsoDepositPUPFBRECO'

photonPFIsolationValuesPFBRECOSequence = (
    phPFIsoValueCharged03PFIdPFBRECO+
    phPFIsoValueChargedAll03PFIdPFBRECO+
    phPFIsoValueGamma03PFIdPFBRECO+
    phPFIsoValueNeutral03PFIdPFBRECO+
    phPFIsoValuePU03PFIdPFBRECO+
    ##############################
    phPFIsoValueCharged04PFIdPFBRECO+
    phPFIsoValueChargedAll04PFIdPFBRECO+
    phPFIsoValueGamma04PFIdPFBRECO+
    phPFIsoValueNeutral04PFIdPFBRECO+
    phPFIsoValuePU04PFIdPFBRECO
    )

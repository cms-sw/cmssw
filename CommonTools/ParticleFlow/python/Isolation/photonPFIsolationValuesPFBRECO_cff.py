import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.photonPFIsolationValues_cff import *

phPFIsoValueCharged03PFIdPFBRECO = phPFIsoValueCharged03PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositChargedPFBRECO')}
)
phPFIsoValueChargedAll03PFIdPFBRECO = phPFIsoValueChargedAll03PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositChargedAllPFBRECO')}
)
phPFIsoValueGamma03PFIdPFBRECO = phPFIsoValueGamma03PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositGammaPFBRECO')}
)
phPFIsoValueNeutral03PFIdPFBRECO = phPFIsoValueNeutral03PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositNeutralPFBRECO')}
)
phPFIsoValuePU03PFIdPFBRECO = phPFIsoValuePU03PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositPUPFBRECO')}
)
phPFIsoValueCharged04PFIdPFBRECO = phPFIsoValueCharged04PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositChargedPFBRECO')}
)
phPFIsoValueChargedAll04PFIdPFBRECO = phPFIsoValueChargedAll04PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositChargedAllPFBRECO')}
)
phPFIsoValueGamma04PFIdPFBRECO = phPFIsoValueGamma04PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositGammaPFBRECO')}
)
phPFIsoValueNeutral04PFIdPFBRECO = phPFIsoValueNeutral04PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositNeutralPFBRECO')}
)
phPFIsoValuePU04PFIdPFBRECO = phPFIsoValuePU04PFId.clone(
    deposits = {0: dict(src = 'phPFIsoDepositPUPFBRECO')}
)
photonPFIsolationValuesPFBRECOTask = cms.Task(
    phPFIsoValueCharged03PFIdPFBRECO,
    phPFIsoValueChargedAll03PFIdPFBRECO,
    phPFIsoValueGamma03PFIdPFBRECO,
    phPFIsoValueNeutral03PFIdPFBRECO,
    phPFIsoValuePU03PFIdPFBRECO,
    ##############################
    phPFIsoValueCharged04PFIdPFBRECO,
    phPFIsoValueChargedAll04PFIdPFBRECO,
    phPFIsoValueGamma04PFIdPFBRECO,
    phPFIsoValueNeutral04PFIdPFBRECO,
    phPFIsoValuePU04PFIdPFBRECO
    )


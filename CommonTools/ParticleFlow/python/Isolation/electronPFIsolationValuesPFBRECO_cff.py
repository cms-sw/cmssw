import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *

elPFIsoValueCharged03PFIdPFBRECO = elPFIsoValueCharged03PFId.clone(
    deposits = {0: dict(src = 'elPFIsoDepositChargedPFBRECO')}
)
elPFIsoValueChargedAll03PFIdPFBRECO = elPFIsoValueChargedAll03PFId.clone(
    deposits = {0: dict(src = 'elPFIsoDepositChargedAllPFBRECO')}
)
elPFIsoValueGamma03PFIdPFBRECO = elPFIsoValueGamma03PFId.clone(
    deposits = {0: dict(src = 'elPFIsoDepositGammaPFBRECO')}
)
elPFIsoValueNeutral03PFIdPFBRECO = elPFIsoValueNeutral03PFId.clone(
    deposits = {0: dict(src = 'elPFIsoDepositNeutralPFBRECO')}
)
elPFIsoValuePU03PFIdPFBRECO = elPFIsoValuePU03PFId.clone(
    deposits = {0: dict(src = 'elPFIsoDepositPUPFBRECO')}
)
elPFIsoValueCharged04PFIdPFBRECO = elPFIsoValueCharged03PFIdPFBRECO.clone(
    deposits = {0: dict(deltaR = 0.4)}
)
elPFIsoValueChargedAll04PFIdPFBRECO = elPFIsoValueChargedAll03PFIdPFBRECO.clone(
    deposits = {0: dict(deltaR = 0.4)}
)
elPFIsoValueGamma04PFIdPFBRECO = elPFIsoValueGamma03PFIdPFBRECO.clone(
    deposits = {0: dict(deltaR = 0.4)}
)
elPFIsoValueNeutral04PFIdPFBRECO = elPFIsoValueNeutral03PFIdPFBRECO.clone(
    deposits = {0: dict(deltaR = 0.4)}
)
elPFIsoValuePU04PFIdPFBRECO = elPFIsoValuePU03PFIdPFBRECO.clone(
    deposits = {0: dict(deltaR = 0.4)}
)
##########Now the PFNoId
elPFIsoValueCharged03NoPFIdPFBRECO     =  elPFIsoValueCharged03PFIdPFBRECO.clone()
elPFIsoValueChargedAll03NoPFIdPFBRECO  =  elPFIsoValueChargedAll03PFIdPFBRECO.clone()
elPFIsoValueGamma03NoPFIdPFBRECO       =  elPFIsoValueGamma03PFIdPFBRECO.clone()
elPFIsoValueNeutral03NoPFIdPFBRECO     =  elPFIsoValueNeutral03PFIdPFBRECO.clone()
elPFIsoValuePU03NoPFIdPFBRECO          =  elPFIsoValuePU03PFIdPFBRECO.clone()
# Customization - No longer needed with new recommendation
#elPFIsoValueCharged03NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueChargedAll03NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValuePU03NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueGamma03NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')


elPFIsoValueCharged04NoPFIdPFBRECO     =  elPFIsoValueCharged04PFIdPFBRECO.clone()
elPFIsoValueChargedAll04NoPFIdPFBRECO  =  elPFIsoValueChargedAll04PFIdPFBRECO.clone()
elPFIsoValueGamma04NoPFIdPFBRECO       =  elPFIsoValueGamma04PFIdPFBRECO.clone()
elPFIsoValueNeutral04NoPFIdPFBRECO     =  elPFIsoValueNeutral04PFIdPFBRECO.clone()
elPFIsoValuePU04NoPFIdPFBRECO          =  elPFIsoValuePU04PFIdPFBRECO.clone()
#elPFIsoValueCharged04NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueChargedAll04NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValuePU04NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueGamma04NoPFIdPFBRECO.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')

electronPFIsolationValuesPFBRECOTask = cms.Task(
    elPFIsoValueCharged03PFIdPFBRECO,
    elPFIsoValueChargedAll03PFIdPFBRECO,
    elPFIsoValueGamma03PFIdPFBRECO,
    elPFIsoValueNeutral03PFIdPFBRECO,
    elPFIsoValuePU03PFIdPFBRECO,
    ##############################
    elPFIsoValueCharged04PFIdPFBRECO,
    elPFIsoValueChargedAll04PFIdPFBRECO,
    elPFIsoValueGamma04PFIdPFBRECO,
    elPFIsoValueNeutral04PFIdPFBRECO,
    elPFIsoValuePU04PFIdPFBRECO,
    ##############################
    elPFIsoValueCharged03NoPFIdPFBRECO,
    elPFIsoValueChargedAll03NoPFIdPFBRECO,
    elPFIsoValueGamma03NoPFIdPFBRECO,
    elPFIsoValueNeutral03NoPFIdPFBRECO,
    elPFIsoValuePU03NoPFIdPFBRECO,
    ##############################
    elPFIsoValueCharged04NoPFIdPFBRECO,
    elPFIsoValueChargedAll04NoPFIdPFBRECO,
    elPFIsoValueGamma04NoPFIdPFBRECO,
    elPFIsoValueNeutral04NoPFIdPFBRECO,
    elPFIsoValuePU04NoPFIdPFBRECO)

electronPFIsolationValuesPFBRECOSequence = cms.Sequence(electronPFIsolationValuesPFBRECOTask)

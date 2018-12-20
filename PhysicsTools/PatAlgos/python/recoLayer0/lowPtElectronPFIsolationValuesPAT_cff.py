import FWCore.ParameterSet.Config as cms

import CommonTools.ParticleFlow.Isolation.electronPFIsolationValuesPFBRECO_cff as _m

#
# TODO: to be discussed if a veto cone is now needed given lowPtElectrons are not in PF
#

lowPtElPFIsoValueCharged03PFIdPAT = _m.elPFIsoValueCharged03PFIdPFBRECO.clone()
lowPtElPFIsoValueCharged03PFIdPAT.deposits[0].src = 'lowPtElPFIsoDepositChargedPAT'

lowPtElPFIsoValueChargedAll03PFIdPAT = _m.elPFIsoValueChargedAll03PFIdPFBRECO.clone()
lowPtElPFIsoValueChargedAll03PFIdPAT.deposits[0].src = 'lowPtElPFIsoDepositChargedAllPAT'

lowPtElPFIsoValueGamma03PFIdPAT = _m.elPFIsoValueGamma03PFIdPFBRECO.clone()
lowPtElPFIsoValueGamma03PFIdPAT.deposits[0].src = 'lowPtElPFIsoDepositGammaPAT'

lowPtElPFIsoValueNeutral03PFIdPAT = _m.elPFIsoValueNeutral03PFIdPFBRECO.clone()
lowPtElPFIsoValueNeutral03PFIdPAT.deposits[0].src = 'lowPtElPFIsoDepositNeutralPAT'

lowPtElPFIsoValuePU03PFIdPAT = _m.elPFIsoValuePU03PFIdPFBRECO.clone()
lowPtElPFIsoValuePU03PFIdPAT.deposits[0].src = 'lowPtElPFIsoDepositPUPAT'

lowPtElPFIsoValueCharged04PFIdPAT = lowPtElPFIsoValueCharged03PFIdPAT.clone()
lowPtElPFIsoValueCharged04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

lowPtElPFIsoValueChargedAll04PFIdPAT = lowPtElPFIsoValueChargedAll03PFIdPAT.clone()
lowPtElPFIsoValueChargedAll04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

lowPtElPFIsoValueGamma04PFIdPAT = lowPtElPFIsoValueGamma03PFIdPAT.clone()
lowPtElPFIsoValueGamma04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

lowPtElPFIsoValueNeutral04PFIdPAT = lowPtElPFIsoValueNeutral03PFIdPAT.clone()
lowPtElPFIsoValueNeutral04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

lowPtElPFIsoValuePU04PFIdPAT = lowPtElPFIsoValuePU03PFIdPAT.clone()
lowPtElPFIsoValuePU04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

##########Now the PFNoId
lowPtElPFIsoValueCharged03NoPFIdPAT     =  lowPtElPFIsoValueCharged03PFIdPAT.clone()
lowPtElPFIsoValueChargedAll03NoPFIdPAT  =  lowPtElPFIsoValueChargedAll03PFIdPAT.clone()
lowPtElPFIsoValueGamma03NoPFIdPAT       =  lowPtElPFIsoValueGamma03PFIdPAT.clone()
lowPtElPFIsoValueNeutral03NoPFIdPAT     =  lowPtElPFIsoValueNeutral03PFIdPAT.clone()
lowPtElPFIsoValuePU03NoPFIdPAT          =  lowPtElPFIsoValuePU03PFIdPAT.clone()
# Customization - No longer needed with new recommendation
#lowPtElPFIsoValueCharged03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#lowPtElPFIsoValueChargedAll03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#lowPtElPFIsoValuePU03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#lowPtElPFIsoValueGamma03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')


lowPtElPFIsoValueCharged04NoPFIdPAT     =  lowPtElPFIsoValueCharged04PFIdPAT.clone()
lowPtElPFIsoValueChargedAll04NoPFIdPAT  =  lowPtElPFIsoValueChargedAll04PFIdPAT.clone()
lowPtElPFIsoValueGamma04NoPFIdPAT       =  lowPtElPFIsoValueGamma04PFIdPAT.clone()
lowPtElPFIsoValueNeutral04NoPFIdPAT     =  lowPtElPFIsoValueNeutral04PFIdPAT.clone()
lowPtElPFIsoValuePU04NoPFIdPAT          =  lowPtElPFIsoValuePU04PFIdPAT.clone()
#lowPtElPFIsoValueCharged04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#lowPtElPFIsoValueChargedAll04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#lowPtElPFIsoValuePU04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#lowPtElPFIsoValueGamma04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')

lowPtElectronPFIsolationValuesPATTask = cms.Task(
    lowPtElPFIsoValueCharged03PFIdPAT,
    lowPtElPFIsoValueChargedAll03PFIdPAT,
    lowPtElPFIsoValueGamma03PFIdPAT,
    lowPtElPFIsoValueNeutral03PFIdPAT,
    lowPtElPFIsoValuePU03PFIdPAT,
    ##############################
    lowPtElPFIsoValueCharged04PFIdPAT,
    lowPtElPFIsoValueChargedAll04PFIdPAT,
    lowPtElPFIsoValueGamma04PFIdPAT,
    lowPtElPFIsoValueNeutral04PFIdPAT,
    lowPtElPFIsoValuePU04PFIdPAT,
    ##############################
    lowPtElPFIsoValueCharged03NoPFIdPAT,
    lowPtElPFIsoValueChargedAll03NoPFIdPAT,
    lowPtElPFIsoValueGamma03NoPFIdPAT,
    lowPtElPFIsoValueNeutral03NoPFIdPAT,
    lowPtElPFIsoValuePU03NoPFIdPAT,
    ##############################
    lowPtElPFIsoValueCharged04NoPFIdPAT,
    lowPtElPFIsoValueChargedAll04NoPFIdPAT,
    lowPtElPFIsoValueGamma04NoPFIdPAT,
    lowPtElPFIsoValueNeutral04NoPFIdPAT,
    lowPtElPFIsoValuePU04NoPFIdPAT)

lowPtElectronPFIsolationValuesPATSequence = cms.Sequence(lowPtElectronPFIsolationValuesPATTask)

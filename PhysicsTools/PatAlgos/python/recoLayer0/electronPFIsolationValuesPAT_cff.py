import FWCore.ParameterSet.Config as cms



elPFIsoValueCharged03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositChargedPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)

elPFIsoValueChargedAll03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositChargedAllPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

elPFIsoValueGamma03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositGammaPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

elPFIsoValueNeutral03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositNeutralPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
        )
    )

elPFIsoValuePU03PFIdPAT = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositPUPAT"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
   )
)



elPFIsoValueCharged04PFIdPAT = elPFIsoValueCharged03PFIdPAT.clone()
elPFIsoValueCharged04PFIdPAT.deposits[0].deltaR = cms.double(0.4)


elPFIsoValueChargedAll04PFIdPAT = elPFIsoValueChargedAll03PFIdPAT.clone()
elPFIsoValueChargedAll04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

elPFIsoValueGamma04PFIdPAT = elPFIsoValueGamma03PFIdPAT.clone()
elPFIsoValueGamma04PFIdPAT.deposits[0].deltaR = cms.double(0.4)


elPFIsoValueNeutral04PFIdPAT = elPFIsoValueNeutral03PFIdPAT.clone()
elPFIsoValueNeutral04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

elPFIsoValuePU04PFIdPAT = elPFIsoValuePU03PFIdPAT.clone()
elPFIsoValuePU04PFIdPAT.deposits[0].deltaR = cms.double(0.4)

##########Now the PFNoId
elPFIsoValueCharged03NoPFIdPAT     =  elPFIsoValueCharged03PFIdPAT.clone()
elPFIsoValueChargedAll03NoPFIdPAT  =  elPFIsoValueChargedAll03PFIdPAT.clone()
elPFIsoValueGamma03NoPFIdPAT       =  elPFIsoValueGamma03PFIdPAT.clone()
elPFIsoValueNeutral03NoPFIdPAT     =  elPFIsoValueNeutral03PFIdPAT.clone()
elPFIsoValuePU03NoPFIdPAT          =  elPFIsoValuePU03PFIdPAT.clone()
# Customization - No longer needed with new recommendation
#elPFIsoValueCharged03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueChargedAll03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValuePU03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueGamma03NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')


elPFIsoValueCharged04NoPFIdPAT     =  elPFIsoValueCharged04PFIdPAT.clone()
elPFIsoValueChargedAll04NoPFIdPAT  =  elPFIsoValueChargedAll04PFIdPAT.clone()
elPFIsoValueGamma04NoPFIdPAT       =  elPFIsoValueGamma04PFIdPAT.clone()
elPFIsoValueNeutral04NoPFIdPAT     =  elPFIsoValueNeutral04PFIdPAT.clone()
elPFIsoValuePU04NoPFIdPAT          =  elPFIsoValuePU04PFIdPAT.clone()
#elPFIsoValueCharged04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueChargedAll04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValuePU04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueGamma04NoPFIdPAT.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')

electronPFIsolationValuesPATSequence = (
    elPFIsoValueCharged03PFIdPAT+
    elPFIsoValueChargedAll03PFIdPAT+
    elPFIsoValueGamma03PFIdPAT+
    elPFIsoValueNeutral03PFIdPAT+
    elPFIsoValuePU03PFIdPAT+
    ##############################
    elPFIsoValueCharged04PFIdPAT+
    elPFIsoValueChargedAll04PFIdPAT+
    elPFIsoValueGamma04PFIdPAT+
    elPFIsoValueNeutral04PFIdPAT+
    elPFIsoValuePU04PFIdPAT+
    ##############################
    elPFIsoValueCharged03NoPFIdPAT+
    elPFIsoValueChargedAll03NoPFIdPAT+
    elPFIsoValueGamma03NoPFIdPAT+
    elPFIsoValueNeutral03NoPFIdPAT+
    elPFIsoValuePU03NoPFIdPAT+
    ##############################
    elPFIsoValueCharged04NoPFIdPAT+
    elPFIsoValueChargedAll04NoPFIdPAT+
    elPFIsoValueGamma04NoPFIdPAT+
    elPFIsoValueNeutral04NoPFIdPAT+
    elPFIsoValuePU04NoPFIdPAT)

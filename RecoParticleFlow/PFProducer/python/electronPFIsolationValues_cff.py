import FWCore.ParameterSet.Config as cms



elPFIsoValueCharged03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
     )
)

elPFIsoValueChargedAll03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
     )
   )
)

elPFIsoValueGamma03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
      )
   )
)

elPFIsoValueNeutral03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring(),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
        )
    )

elPFIsoValuePU03PFId = cms.EDProducer("PFCandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("elPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum'),
            PivotCoordinatesForEBEE = cms.bool(True)
            )
   )
)



elPFIsoValueCharged04PFId = elPFIsoValueCharged03PFId.clone()
elPFIsoValueCharged04PFId.deposits[0].deltaR = cms.double(0.4)


elPFIsoValueChargedAll04PFId = elPFIsoValueChargedAll03PFId.clone()
elPFIsoValueChargedAll04PFId.deposits[0].deltaR = cms.double(0.4)

elPFIsoValueGamma04PFId = elPFIsoValueGamma03PFId.clone()
elPFIsoValueGamma04PFId.deposits[0].deltaR = cms.double(0.4)


elPFIsoValueNeutral04PFId = elPFIsoValueNeutral03PFId.clone()
elPFIsoValueNeutral04PFId.deposits[0].deltaR = cms.double(0.4)

elPFIsoValuePU04PFId = elPFIsoValuePU03PFId.clone()
elPFIsoValuePU04PFId.deposits[0].deltaR = cms.double(0.4)

##########Now the PFNoId
elPFIsoValueCharged03NoPFId     =  elPFIsoValueCharged03PFId.clone()           
elPFIsoValueChargedAll03NoPFId  =  elPFIsoValueChargedAll03PFId.clone()
elPFIsoValueGamma03NoPFId       =  elPFIsoValueGamma03PFId.clone()         
elPFIsoValueNeutral03NoPFId     =  elPFIsoValueNeutral03PFId.clone()       
elPFIsoValuePU03NoPFId          =  elPFIsoValuePU03PFId.clone()            
# Customization - No longer needed with new recommendation  
#elPFIsoValueCharged03NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueChargedAll03NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValuePU03NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)') 
#elPFIsoValueGamma03NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')


elPFIsoValueCharged04NoPFId     =  elPFIsoValueCharged04PFId.clone()       
elPFIsoValueChargedAll04NoPFId  =  elPFIsoValueChargedAll04PFId.clone()    
elPFIsoValueGamma04NoPFId       =  elPFIsoValueGamma04PFId.clone()         
elPFIsoValueNeutral04NoPFId     =  elPFIsoValueNeutral04PFId.clone()       
elPFIsoValuePU04NoPFId          =  elPFIsoValuePU04PFId.clone()            
#elPFIsoValueCharged04NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValueChargedAll04NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)')
#elPFIsoValuePU04NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:ConeVeto(0.015)','EcalEndcaps:ConeVeto(0.015)') 
#elPFIsoValueGamma04NoPFId.deposits[0].vetos = cms.vstring('EcalBarrel:RectangularEtaPhiVeto(-0.02,0.02,-0.5,0.5)','EcalEndcaps:ConeVeto(0.08)')

electronPFIsolationValuesSequence = (
    elPFIsoValueCharged03PFId+
    elPFIsoValueChargedAll03PFId+
    elPFIsoValueGamma03PFId+
    elPFIsoValueNeutral03PFId+
    elPFIsoValuePU03PFId+
    ############################## 
    elPFIsoValueCharged04PFId+
    elPFIsoValueChargedAll04PFId+
    elPFIsoValueGamma04PFId+
    elPFIsoValueNeutral04PFId+
    elPFIsoValuePU04PFId+
    ############################## 
    elPFIsoValueCharged03NoPFId+
    elPFIsoValueChargedAll03NoPFId+
    elPFIsoValueGamma03NoPFId+
    elPFIsoValueNeutral03NoPFId+
    elPFIsoValuePU03NoPFId+
    ############################## 
    elPFIsoValueCharged04NoPFId+
    elPFIsoValueChargedAll04NoPFId+
    elPFIsoValueGamma04NoPFId+
    elPFIsoValueNeutral04NoPFId+
    elPFIsoValuePU04NoPFId)

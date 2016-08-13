import FWCore.ParameterSet.Config as cms

hcalRawDatauHTR = cms.EDProducer("HcalDigiToRawuHTR",
   ElectronicsMap = cms.string(""),
   QIE10 = cms.InputTag("simHcalDigis", "HFQIE10DigiCollection"),
   QIE11 = cms.InputTag("simHcalDigis", "HBHEQIE11DigiCollection"),
   HBHEqie8 = cms.InputTag("simHcalDigis"),
   HFqie8 = cms.InputTag("simHcalDigis"),
   TP = cms.InputTag("simHcalTriggerPrimitiveDigis"),
)


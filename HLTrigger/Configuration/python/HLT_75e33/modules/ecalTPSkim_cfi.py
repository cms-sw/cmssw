import FWCore.ParameterSet.Config as cms

ecalTPSkim = cms.EDProducer("EcalTPSkimmer",
    chStatusToSelectTP = cms.vuint32(13),
    doBarrel = cms.bool(True),
    doEndcap = cms.bool(True),
    skipModule = cms.bool(False),
    tpInputCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    tpOutputCollection = cms.string('')
)

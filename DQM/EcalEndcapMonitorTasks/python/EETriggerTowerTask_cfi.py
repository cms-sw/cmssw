import FWCore.ParameterSet.Config as cms

ecalEndcapTriggerTowerTask = cms.EDFilter("EETriggerTowerTask",
    OutputRootFile = cms.untracked.string(''),
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    enableCleanup = cms.untracked.bool(True),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("ecalTriggerPrimitiveDigis")
)



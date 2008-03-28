import FWCore.ParameterSet.Config as cms

ecalBarrelTriggerTowerTask = cms.EDFilter("EBTriggerTowerTask",
    OutputRootFile = cms.untracked.string(''),
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    enableCleanup = cms.untracked.bool(True),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("ecalTriggerPrimitiveDigis")
)



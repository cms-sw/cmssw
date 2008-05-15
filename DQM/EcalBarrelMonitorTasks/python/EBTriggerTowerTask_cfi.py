import FWCore.ParameterSet.Config as cms

ecalBarrelTriggerTowerTask = cms.EDFilter("EBTriggerTowerTask",
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    OutputRootFile = cms.untracked.string(''),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    prefixME = cms.untracked.string('EcalBarrel'),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("ecalTriggerPrimitiveDigis")
)



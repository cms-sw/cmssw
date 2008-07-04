import FWCore.ParameterSet.Config as cms

ecalBarrelTriggerTowerTask = cms.EDFilter("EBTriggerTowerTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    OutputRootFile = cms.untracked.string(''),
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("ecalTriggerPrimitiveDigis")
)


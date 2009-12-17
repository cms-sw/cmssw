import FWCore.ParameterSet.Config as cms

ecalBarrelTriggerTowerTask = cms.EDAnalyzer("EBTriggerTowerTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    OutputRootFile = cms.untracked.string(''),
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis")
)


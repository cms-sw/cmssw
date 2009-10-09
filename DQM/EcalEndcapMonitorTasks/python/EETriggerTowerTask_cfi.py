import FWCore.ParameterSet.Config as cms

ecalEndcapTriggerTowerTask = cms.EDAnalyzer("EETriggerTowerTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    OutputRootFile = cms.untracked.string(''),
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis")
)


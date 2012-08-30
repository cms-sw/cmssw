import FWCore.ParameterSet.Config as cms

ecalBarrelTriggerTowerTask = cms.EDAnalyzer("EBTriggerTowerTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    OutputRootFile = cms.untracked.string(''),
    EcalTrigPrimDigiCollectionReal = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EcalTrigPrimDigiCollectionEmul = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    HLTResultsCollection = cms.InputTag("TriggerResults"),
    HLTCaloHLTBit = cms.untracked.string('HLT_EgammaSuperClusterOnly_L1R'),
    HLTMuonHLTBit = cms.untracked.string('HLT_L1MuOpen')
)


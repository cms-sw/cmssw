import FWCore.ParameterSet.Config as cms

ecalBarrelClusterTask = cms.EDAnalyzer("EBClusterTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    BasicClusterCollection = cms.InputTag("particleFlowClusterECAL"),
    SuperClusterCollection = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


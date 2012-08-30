import FWCore.ParameterSet.Config as cms

ecalBarrelClusterTask = cms.EDAnalyzer("EBClusterTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalDigis"),
    BasicClusterCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    SuperClusterCollection = cms.InputTag("correctedHybridSuperClusters"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB")
)


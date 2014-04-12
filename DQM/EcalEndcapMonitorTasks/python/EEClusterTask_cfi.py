import FWCore.ParameterSet.Config as cms

ecalEndcapClusterTask = cms.EDAnalyzer("EEClusterTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    BasicClusterCollection = cms.InputTag("particleFlowClusterECAL"),
    SuperClusterCollection = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE")
)


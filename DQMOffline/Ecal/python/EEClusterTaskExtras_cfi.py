import FWCore.ParameterSet.Config as cms

ecalEndcapClusterTaskExtras = cms.EDAnalyzer("EEClusterTaskExtras",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    SuperClusterCollection = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    l1GlobalReadoutRecord = cms.InputTag('gtDigis'),
    l1GlobalMuonReadoutRecord = cms.InputTag("gtDigis")
)

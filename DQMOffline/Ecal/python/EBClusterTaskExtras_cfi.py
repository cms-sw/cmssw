import FWCore.ParameterSet.Config as cms

ecalBarrelClusterTaskExtras = cms.EDAnalyzer("EBClusterTaskExtras",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    SuperClusterCollection = cms.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    l1GlobalReadoutRecord = cms.InputTag('gtDigis'),
    l1GlobalMuonReadoutRecord = cms.InputTag("gtDigis")
)


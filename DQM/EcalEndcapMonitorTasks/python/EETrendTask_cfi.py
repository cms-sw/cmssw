import FWCore.ParameterSet.Config as cms

ecalEndcapTrendTask = cms.EDAnalyzer("EETrendTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    EEDigiCollection = cms.InputTag("ecalEBunpacker","eeDigis"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    BasicClusterCollection = cms.InputTag("multi5x5BasicClusters","multi5x5EndcapBasicClusters"),
    SuperClusterCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters"),
    FEDRawDataCollection = cms.InputTag("source")
)


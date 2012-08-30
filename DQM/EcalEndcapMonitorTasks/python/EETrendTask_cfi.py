import FWCore.ParameterSet.Config as cms

ecalEndcapTrendTask = cms.EDAnalyzer("EETrendTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    EEDigiCollection = cms.InputTag("ecalDigis","eeDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    BasicClusterCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters"),
    SuperClusterCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapSuperClusters"),
    EEDetIdCollection0 = cms.InputTag("ecalDigis","EcalIntegrityDCCSizeErrors"),
    EEDetIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    EEDetIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    EEDetIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EESRFlagCollection = cms.InputTag("ecalDigis")
)


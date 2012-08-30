import FWCore.ParameterSet.Config as cms

ecalBarrelTrendTask = cms.EDAnalyzer("EBTrendTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    verbose = cms.untracked.bool(False),
    EBDigiCollection = cms.InputTag("ecalDigis","ebDigis"),
    EcalPnDiodeDigiCollection = cms.InputTag("ecalDigis"),
    EcalTrigPrimDigiCollection = cms.InputTag("ecalDigis","EcalTriggerPrimitives"),
    EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    BasicClusterCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters"),
    SuperClusterCollection = cms.InputTag("correctedHybridSuperClusters"),
    EBDetIdCollection0 = cms.InputTag("ecalDigis","EcalIntegrityDCCSizeErrors"),
    EBDetIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    EBDetIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    EBDetIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalDigis","EcalIntegrityMemGainErrors"),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EBSRFlagCollection = cms.InputTag("ecalDigis")
)


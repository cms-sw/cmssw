import FWCore.ParameterSet.Config as cms

ecalBarrelHltTask = cms.EDAnalyzer("EBHltTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    folderName = cms.untracked.string('FEDIntegrity'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EBDetIdCollection0 = cms.InputTag("ecalEBunpacker","EcalIntegrityDCCSizeErrors"),
    EBDetIdCollection1 = cms.InputTag("ecalEBunpacker","EcalIntegrityGainErrors"),
    EBDetIdCollection2 = cms.InputTag("ecalEBunpacker","EcalIntegrityChIdErrors"),
    EBDetIdCollection3 = cms.InputTag("ecalEBunpacker","EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalEBunpacker","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalEBunpacker","EcalIntegrityBlockSizeErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemBlockSizeErrors"),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemGainErrors")
)


import FWCore.ParameterSet.Config as cms

ecalBarrelHltTask = cms.EDAnalyzer("EBHltTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    folderName = cms.untracked.string('FEDIntegrity'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EBDetIdCollection0 = cms.InputTag("ecalDigis","EcalIntegrityDCCSizeErrors"),
    EBDetIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    EBDetIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    EBDetIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalDigis","EcalIntegrityMemGainErrors")
)


import FWCore.ParameterSet.Config as cms

ecalEndcapHltTask = cms.EDAnalyzer("EEHltTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    folderName = cms.untracked.string('FEDIntegrity'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EEDetIdCollection0 = cms.InputTag("ecalDigis","EcalIntegrityDCCSizeErrors"),
    EEDetIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityGainErrors"),
    EEDetIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityChIdErrors"),
    EEDetIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalDigis","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalDigis","EcalIntegrityBlockSizeErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalDigis","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalDigis","EcalIntegrityMemBlockSizeErrors"),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalDigis","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalDigis","EcalIntegrityMemGainErrors")
)


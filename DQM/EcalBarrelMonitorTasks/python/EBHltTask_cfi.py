import FWCore.ParameterSet.Config as cms

ecalBarrelHltTask = cms.EDAnalyzer("EBHltTask",
    folderName = cms.untracked.string('FEDIntegrity'),
    FEDRawDataCollection = cms.InputTag("rawDataCollector"),
    EBDetIdCollection1 = cms.InputTag("ecalDigis", "EcalIntegrityGainErrors"),
    EBDetIdCollection2 = cms.InputTag("ecalDigis", "EcalIntegrityChIdErrors"),
    EBDetIdCollection3 = cms.InputTag("ecalDigis", "EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalDigis", "EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalDigis", "EcalIntegrityBlockSizeErrors")
)

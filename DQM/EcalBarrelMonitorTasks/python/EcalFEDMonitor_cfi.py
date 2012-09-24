import FWCore.ParameterSet.Config as cms

ecalFEDMonitor = cms.EDAnalyzer("EcalFEDMonitor",
    folderName = cms.untracked.string("FEDIntegrity"),
    FEDRawDataTag = cms.untracked.InputTag("rawDataCollector"),
    gainErrorsTag = cms.untracked.InputTag("ecalDigis", "EcalIntegrityGainErrors"),
    chIdErrorsTag = cms.untracked.InputTag("ecalDigis", "EcalIntegrityChIdErrors"),
    gainSwitchErrorsTag = cms.untracked.InputTag("ecalDigis", "EcalIntegrityGainSwitchErrors"),
    towerIdErrorsTag = cms.untracked.InputTag("ecalDigis", "EcalIntegrityTTIdErrors"),
    blockSizeErrorsTag = cms.untracked.InputTag("ecalDigis", "EcalIntegrityBlockSizeErrors")
)

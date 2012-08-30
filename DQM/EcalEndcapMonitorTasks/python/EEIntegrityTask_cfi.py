import FWCore.ParameterSet.Config as cms

ecalEndcapIntegrityTask = cms.EDAnalyzer("EEIntegrityTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
                                         subfolder = cms.untracked.string(''),
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


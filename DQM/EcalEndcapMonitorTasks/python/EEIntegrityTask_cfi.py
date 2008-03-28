import FWCore.ParameterSet.Config as cms

ecalEndcapIntegrityTask = cms.EDFilter("EEIntegrityTask",
    enableCleanup = cms.untracked.bool(True),
    EEDetIdCollection0 = cms.InputTag("ecalEBunpacker","EcalIntegrityDCCSizeErrors"),
    EEDetIdCollection1 = cms.InputTag("ecalEBunpacker","EcalIntegrityGainErrors"),
    EEDetIdCollection2 = cms.InputTag("ecalEBunpacker","EcalIntegrityChIdErrors"),
    EEDetIdCollection3 = cms.InputTag("ecalEBunpacker","EcalIntegrityGainSwitchErrors"),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemBlockSize"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemGainErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalEBunpacker","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalEBunpacker","EcalIntegrityBlockSizeErrors")
)



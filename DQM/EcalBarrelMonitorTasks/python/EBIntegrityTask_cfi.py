import FWCore.ParameterSet.Config as cms

ecalBarrelIntegrityTask = cms.EDFilter("EBIntegrityTask",
    EBDetIdCollection3 = cms.InputTag("ecalEBunpacker","EcalIntegrityGainSwitchErrors"),
    EBDetIdCollection2 = cms.InputTag("ecalEBunpacker","EcalIntegrityChIdErrors"),
    EBDetIdCollection1 = cms.InputTag("ecalEBunpacker","EcalIntegrityGainErrors"),
    EBDetIdCollection0 = cms.InputTag("ecalEBunpacker","EcalIntegrityDCCSizeErrors"),
    enableCleanup = cms.untracked.bool(True),
    EcalElectronicsIdCollection5 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemChIdErrors"),
    EcalElectronicsIdCollection4 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemBlockSize"),
    EcalElectronicsIdCollection6 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemGainErrors"),
    EcalElectronicsIdCollection1 = cms.InputTag("ecalEBunpacker","EcalIntegrityTTIdErrors"),
    EcalElectronicsIdCollection3 = cms.InputTag("ecalEBunpacker","EcalIntegrityMemTtIdErrors"),
    EcalElectronicsIdCollection2 = cms.InputTag("ecalEBunpacker","EcalIntegrityBlockSizeErrors")
)



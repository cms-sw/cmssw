import FWCore.ParameterSet.Config as cms

ecalBarrelIntegrityTask = cms.EDAnalyzer("EBIntegrityTask",
    prefixME = cms.untracked.string('EcalBarrel'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
                                         subfolder = cms.untracked.string(""),
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


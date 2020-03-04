import FWCore.ParameterSet.Config as cms

# Dec 2019: ecalDQMCollectionTags was changed from an untracked PSet to a tracked PSet.
# The reason is that this PSet is part of the offline DQM configuration for both pp
# and HI runs. For HI runs, there is a function in the the config builder that
# replaces all inputTags named "rawDataCollector" to "rawDataMapperByLabel", which
# is necessary for HI runs. As of Dec 2019, this function was called "MassReplaceInputTag".
# This only works if the collection tags below are part of a tracked PSet.

ecalDQMCollectionTags = cms.PSet(
    Source = cms.untracked.InputTag("rawDataCollector"),
    EcalRawData = cms.untracked.InputTag("ecalDigis"),
    EBGainErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityGainErrors"),
    EEGainErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityGainErrors"),
    EBChIdErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityChIdErrors"),
    EEChIdErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityChIdErrors"),
    EBGainSwitchErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityGainSwitchErrors"),
    EEGainSwitchErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityGainSwitchErrors"),
    TowerIdErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityTTIdErrors"),
    BlockSizeErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityBlockSizeErrors"),
    MEMTowerIdErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityMemTtIdErrors"),
    MEMBlockSizeErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityMemBlockSizeErrors"),
    MEMChIdErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityMemChIdErrors"),
    MEMGainErrors = cms.untracked.InputTag("ecalDigis", "EcalIntegrityMemGainErrors"),
    EBSrFlag = cms.untracked.InputTag("ecalDigis"),
    EESrFlag = cms.untracked.InputTag("ecalDigis"),
    EBDigi = cms.untracked.InputTag("ecalDigis", "ebDigis"),
    EEDigi = cms.untracked.InputTag("ecalDigis", "eeDigis"),
    PnDiodeDigi = cms.untracked.InputTag("ecalDigis"),
    TrigPrimDigi = cms.untracked.InputTag("ecalDigis", "EcalTriggerPrimitives"),
    TrigPrimEmulDigi = cms.untracked.InputTag("valEcalTriggerPrimitiveDigis"),
    EBUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit", "EcalUncalibRecHitsEB"),
    EEUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit", "EcalUncalibRecHitsEE"),
#    EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalLaserLedUncalibRecHit", "EcalUncalibRecHitsEB"),
#    EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalLaserLedUncalibRecHit", "EcalUncalibRecHitsEE"),
    EBLaserLedUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit", "EcalUncalibRecHitsEB"),
    EELaserLedUncalibRecHit = cms.untracked.InputTag("ecalMultiFitUncalibRecHit", "EcalUncalibRecHitsEE"),
    EBTestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit", "EcalUncalibRecHitsEB"),
    EETestPulseUncalibRecHit = cms.untracked.InputTag("ecalTestPulseUncalibRecHit", "EcalUncalibRecHitsEE"),
    EBRecHit = cms.untracked.InputTag("ecalRecHit", "EcalRecHitsEB"),
    EERecHit = cms.untracked.InputTag("ecalRecHit", "EcalRecHitsEE"),
    EBReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEB"),
    EEReducedRecHit = cms.untracked.InputTag("reducedEcalRecHitsEE"),
    EBBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
    EEBasicCluster = cms.untracked.InputTag("particleFlowClusterECAL"),
    EBSuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALBarrel"),
    EESuperCluster = cms.untracked.InputTag("particleFlowSuperClusterECAL", "particleFlowSuperClusterECALEndcapWithPreshower")
)

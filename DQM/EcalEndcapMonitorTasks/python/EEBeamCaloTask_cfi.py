import FWCore.ParameterSet.Config as cms

ecalEndcapBeamCaloTask = cms.EDFilter("EEBeamCaloTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB")
)


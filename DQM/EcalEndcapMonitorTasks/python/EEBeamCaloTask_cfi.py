import FWCore.ParameterSet.Config as cms

ecalEndcapBeamCaloTask = cms.EDFilter("EEBeamCaloTask",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    enableCleanup = cms.untracked.bool(True)
)



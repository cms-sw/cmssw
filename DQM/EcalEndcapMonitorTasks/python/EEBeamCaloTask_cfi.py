import FWCore.ParameterSet.Config as cms

ecalEndcapBeamCaloTask = cms.EDFilter("EEBeamCaloTask",
    EBDigiCollection = cms.InputTag("ecalEBunpacker","ebDigis"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    enableCleanup = cms.untracked.bool(False),
    prefixME = cms.untracked.string('EcalEndcap')
)



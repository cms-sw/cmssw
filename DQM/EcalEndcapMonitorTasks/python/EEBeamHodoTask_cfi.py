import FWCore.ParameterSet.Config as cms

ecalEndcapBeamHodoTask = cms.EDFilter("EEBeamHodoTask",
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTBHodoscopeRawInfo = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    enableCleanup = cms.untracked.bool(True),
    EcalTBTDCRawInfo = cms.InputTag("ecalEBunpacker"),
    EcalTBHodoscopeRecInfo = cms.InputTag("ecal2006TBHodoscopeReconstructor","EcalTBHodoscopeRecInfo"),
    EcalTBTDCRecInfo = cms.InputTag("ecal2006TBTDCReconstructor","EcalTBTDCRecInfo")
)



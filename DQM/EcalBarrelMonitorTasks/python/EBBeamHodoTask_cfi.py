import FWCore.ParameterSet.Config as cms

ecalBarrelBeamHodoTask = cms.EDFilter("EBBeamHodoTask",
    EcalTBTDCRecInfo = cms.InputTag("ecal2006TBTDCReconstructor","EcalTBTDCRecInfo"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTBHodoscopeRawInfo = cms.InputTag("ecalEBunpacker"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB"),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalTBTDCRawInfo = cms.InputTag("ecalEBunpacker"),
    EcalTBHodoscopeRecInfo = cms.InputTag("ecal2006TBHodoscopeReconstructor","EcalTBHodoscopeRecInfo"),
    prefixME = cms.untracked.string('EcalBarrel')
)



import FWCore.ParameterSet.Config as cms

ecalEndcapBeamHodoTask = cms.EDAnalyzer("EEBeamHodoTask",
    prefixME = cms.untracked.string('EcalEndcap'),
    enableCleanup = cms.untracked.bool(False),
    mergeRuns = cms.untracked.bool(False),    
    EcalRawDataCollection = cms.InputTag("ecalEBunpacker"),
    EcalTBEventHeader = cms.InputTag("ecalEBunpacker"),
    EcalTBTDCRawInfo = cms.InputTag("ecalEBunpacker"),
    EcalTBTDCRecInfo = cms.InputTag("ecal2006TBTDCReconstructor","EcalTBTDCRecInfo"),
    EcalTBHodoscopeRawInfo = cms.InputTag("ecalEBunpacker"),
    EcalTBHodoscopeRecInfo = cms.InputTag("ecal2006TBHodoscopeReconstructor","EcalTBHodoscopeRecInfo"),
    EcalUncalibratedRecHitCollection = cms.InputTag("ecalUncalibHit","EcalUncalibRecHitsEB")
)


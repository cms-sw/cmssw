import FWCore.ParameterSet.Config as cms

tpAnalyzer = cms.EDAnalyzer("EcalTPGAnalyzer",

    TPCollection = cms.InputTag("ecalEBunpacker","EcalTriggerPrimitives"),
    TPEmulatorCollection =  cms.InputTag("ecalTriggerPrimitiveDigis",""),
    DigiCollectionEB = cms.InputTag("ecalEBunpacker","ebDigis"),
    DigiCollectionEE = cms.InputTag("ecalEBunpacker","eeDigis"),
    GTRecordCollection = cms.string('gtDigis'),
    TrackMuonCollection = cms.string('globalCosmicMuons1LegBarrelOnly'),
                                    
    Print = cms.bool(True),
    ReadTriggerPrimitives = cms.bool(True),                                    
    UseEndCap = cms.bool(False)

)

import FWCore.ParameterSet.Config as cms

SiPixelLorentzAngleHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoMu11', 
        'HLT_DoubleMu3', 
        'HLT_DoubleMu3_JPsi', 
        'HLT_DoubleMu3_Upsilon', 
        'HLT_DoubleMu7_Z', 
        'HLT_DoubleMu3_SameSign'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path name 
    TriggerResultsTag = cms.InputTag("TriggerResults")
)



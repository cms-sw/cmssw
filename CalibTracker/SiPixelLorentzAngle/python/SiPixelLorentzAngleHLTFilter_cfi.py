import FWCore.ParameterSet.Config as cms

SiPixelLorentzAngleHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonIso', 
        'HLT2MuonNonIso', 
        'HLT2MuonJPsi', 
        'HLT2MuonUpsilon', 
        'HLT2MuonZ', 
        'HLT2MuonSameSign'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path names
    TriggerResultsTag = cms.InputTag("TriggerResults")
)



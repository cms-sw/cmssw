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
    TriggerResultsTag = cms.InputTag("TriggerResults")
)



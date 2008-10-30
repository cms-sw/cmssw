import FWCore.ParameterSet.Config as cms

isoMuonHLT = cms.EDFilter("HLTHighLevel",
    #HLTPaths = cms.vstring('HLT_L1_MuonOpen'),
    #HLTPaths = cms.vstring('HLT_IsoMu11'),
    HLTPaths = cms.vstring('HLT_L1MuOpen','HLT_IsoMu11'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path name
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



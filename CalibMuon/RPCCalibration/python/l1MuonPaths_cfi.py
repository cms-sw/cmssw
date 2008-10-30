import FWCore.ParameterSet.Config as cms

l1MuonHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_L1Mu'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False),#dont  throw except on unknown path name
    TriggerResultsTag = cms.InputTag("TriggerResults")
)



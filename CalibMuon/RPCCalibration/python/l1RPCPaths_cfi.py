import FWCore.ParameterSet.Config as cms

l1RPCHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_L1Mu'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path names
    TriggerResultsTag = cms.InputTag("TriggerResults")
)



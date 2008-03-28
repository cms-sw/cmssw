import FWCore.ParameterSet.Config as cms

l1RPCHLTFilter = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT1MuonLevel1'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)



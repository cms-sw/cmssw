import FWCore.ParameterSet.Config as cms

isoMuonHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('HLT_IsoMu11'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



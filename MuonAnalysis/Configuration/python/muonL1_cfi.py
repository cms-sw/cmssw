import FWCore.ParameterSet.Config as cms

muonL1 = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLT1MuonLevel1'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



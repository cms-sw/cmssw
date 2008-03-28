import FWCore.ParameterSet.Config as cms

hcalminbiasHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLTHcalPhiSym'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



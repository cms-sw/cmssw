import FWCore.ParameterSet.Config as cms

ecalpi0CalibHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLTEcalPi0'),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



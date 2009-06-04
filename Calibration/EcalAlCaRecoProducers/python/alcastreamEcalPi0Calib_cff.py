import FWCore.ParameterSet.Config as cms

ecalpi0CalibHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCa_EcalPi0*'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), 
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



import FWCore.ParameterSet.Config as cms

ecalpi0CalibHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCa_EcalPi0'),
    andOr = cms.bool(True),
    throw = cms.untracked.bool(False), #dont throw except on unknown path name 
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



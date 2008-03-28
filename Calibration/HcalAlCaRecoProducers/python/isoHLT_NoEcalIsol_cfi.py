import FWCore.ParameterSet.Config as cms

isoHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('CandHLTHcalIsolatedTrackNoEcalIsol'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



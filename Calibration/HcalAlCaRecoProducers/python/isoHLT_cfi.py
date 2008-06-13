import FWCore.ParameterSet.Config as cms

isoHLT = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('AlCa_IsoTrack'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
)



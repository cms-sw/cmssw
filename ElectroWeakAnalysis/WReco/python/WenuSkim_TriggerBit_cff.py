import FWCore.ParameterSet.Config as cms

report = cms.EDFilter("HLTrigReport",
    HLTriggerResults = cms.InputTag("TriggerResults")
)

high = cms.EDFilter("HLTHighLevel",
    HLTPaths = cms.vstring('SingleElectron'),
    byName = cms.bool(True),
    andOr = cms.bool(True),
    TriggerResultsTag = cms.InputTag("TriggerResults")
)

WenuHLTPath = cms.Path(report*high)


import FWCore.ParameterSet.Config as cms

l1Filter = cms.EDFilter(
    "L1Filter",
    inputTag = cms.InputTag("gtDigis"),
    useAODRecord = cms.bool(False),
    useFinalDecision = cms.bool(False),
    algorithms = cms.vstring("L1_SingleEG15")
)

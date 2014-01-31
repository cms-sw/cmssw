import FWCore.ParameterSet.Config as cms

FED_test_Analyzer = cms.EDAnalyzer("Phase2TrackerFED_test_Analyzer",
    InputLabel = cms.InputTag("source")
)



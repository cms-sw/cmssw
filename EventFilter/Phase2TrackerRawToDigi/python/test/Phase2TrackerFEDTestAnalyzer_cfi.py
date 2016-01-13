import FWCore.ParameterSet.Config as cms

FEDTestAnalyzer = cms.EDAnalyzer("Phase2TrackerFEDTestAnalyzer",
    InputLabel = cms.InputTag("source")
)



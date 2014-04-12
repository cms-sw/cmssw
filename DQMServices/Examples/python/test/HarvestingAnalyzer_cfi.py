import FWCore.ParameterSet.Config as cms

harvestinganalyzer = cms.EDAnalyzer("HarvestingAnalyzer",
    Verbosity = cms.untracked.int32(0),
    Name = cms.untracked.string('HarvestingAnalyzer')
)


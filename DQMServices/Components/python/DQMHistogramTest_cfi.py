import FWCore.ParameterSet.Config as cms

dqmHistogramTest = cms.EDAnalyzer("DQMHistogramTest",
        path = cms.untracked.string("DQMTest/DBDump"),
	histograms = cms.untracked.vstring(),
)

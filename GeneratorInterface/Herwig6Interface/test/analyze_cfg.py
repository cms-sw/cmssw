import FWCore.ParameterSet.Config as cms

process = cms.Process("Analyze")

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring("file:herwigHZZ4mu.root")
)

process.h4muAnalyzer = cms.EDAnalyzer("H4muExampleAnalyzer")

process.TFileService = cms.Service("TFileService",
	fileName = cms.string("H4mu_histo.root")
)

process.path = cms.Path(
	process.h4muAnalyzer
)

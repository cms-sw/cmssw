import FWCore.ParameterSet.Config as cms

process = cms.Process("runRivetAnalysis")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:gen.root')
)

process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")

process.rivetAnalyzer.AnalysisNames = cms.vstring('CMS_EWK_10_012')

process.p = cms.Path(process.rivetAnalyzer)



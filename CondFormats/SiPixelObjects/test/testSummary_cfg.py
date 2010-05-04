import FWCore.ParameterSet.Config as cms

process = cms.Process("bla")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))


process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('*'),
    destinations = cms.untracked.vstring('read'),
    read = cms.untracked.PSet( threshold = cms.untracked.string('INFO'))
)

process.testSummary =  cms.EDAnalyzer("SiPixelTestSummary")

process.p = cms.Path(process.testSummary)


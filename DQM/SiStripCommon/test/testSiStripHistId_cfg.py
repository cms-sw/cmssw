import FWCore.ParameterSet.Config as cms

process = cms.Process("DBTest")
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.print = cms.OutputModule("AsciiOutputModule")
process.read = cms.EDAnalyzer("testSiStripHistId")

process.p1 = cms.EndPath(process.read+process.print)

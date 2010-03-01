import FWCore.ParameterSet.Config as cms

process = cms.Process("DBTest")
process.source = cms.Source("EmptySource",
    maxEvents = cms.untracked.int32(1)
)

process.print = cms.OutputModule("AsciiOutputModule")

process.read = cms.EDAnalyzer("Test_SiStrip_HistId")

process.p1 = cms.Path(process.read+process.print)



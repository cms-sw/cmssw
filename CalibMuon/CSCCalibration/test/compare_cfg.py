import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.demo = cms.EDAnalyzer('Compare')





process.maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.output = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.demo)
process.ep = cms.EndPath(process.output)

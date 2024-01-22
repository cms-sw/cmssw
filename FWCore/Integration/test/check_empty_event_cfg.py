import FWCore.ParameterSet.Config as cms

process = cms.Process("EMPTY")

process.test = cms.EDAnalyzer("EventContentAnalyzer", listPathStatus = cms.untracked.bool(True))

process.e = cms.EndPath(process.test)

#process.out = cms.OutputModule("AsciiOutputModule")
#process.e2 = cms.EndPath(process.out)

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

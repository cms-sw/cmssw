import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("RepeatingCachedRootSource", fileName = cms.untracked.string("file:PoolInputRepeatingSource.root"), repeatNEvents = cms.untracked.uint32(2))

process.maxEvents.input = 10000

process.checker = cms.EDAnalyzer("OtherThingAnalyzer")
#process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.checker)
#process.o = cms.EndPath(process.dump)

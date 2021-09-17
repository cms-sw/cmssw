import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.source = cms.Source("RepeatingCachedRootSource", fileName = cms.untracked.string("file:PoolInputTest.root"), repeatNEvents = cms.untracked.uint32(2))

process.maxEvents.input = 10000

process.OtherThing = cms.EDProducer("OtherThingProducer")
#process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(process.OtherThing)
#process.o = cms.EndPath(process.dump)

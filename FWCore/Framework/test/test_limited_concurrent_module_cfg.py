import FWCore.ParameterSet.Config as cms

process = cms.Process("LIMITEDCONCURRENT")

process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
                            numberOfStreams = cms.untracked.uint32(6),
                            numberOfThreads = cms.untracked.uint32(6)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20))

process.i1 = cms.EDProducer("BusyWaitIntLimitedProducer", ivalue = cms.int32(1),
                            iterations = cms.uint32(300*1000),
                            concurrencyLimit = cms.untracked.uint32(2))


process.c1 = cms.EDAnalyzer("ConsumingOneSharedResourceAnalyzer",
                            valueMustMatch = cms.untracked.int32(1),
                            moduleLabel = cms.untracked.InputTag("i1"),
                            resourceName = cms.untracked.string("foo"))
                            

process.t = cms.Task(process.i1)

process.p = cms.Path(process.c1, process.t)

process.add_(cms.Service("ConcurrentModuleTimer",
                         modulesToExclude = cms.untracked.vstring("TriggerResults"),
                         excludeSource = cms.untracked.bool(True)))

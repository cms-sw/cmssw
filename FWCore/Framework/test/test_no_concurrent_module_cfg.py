import FWCore.ParameterSet.Config as cms

process = cms.Process("NOCONCURRENT")

process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
                            numberOfStreams = cms.untracked.uint32(4),
                            numberOfThreads = cms.untracked.uint32(4)                            
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(20))

process.i1 = cms.EDProducer("BusyWaitIntLegacyProducer", ivalue = cms.int32(1),
  iterations = cms.uint32(300*1000))
process.i2 = cms.EDProducer("BusyWaitIntLegacyProducer", ivalue = cms.int32(2),
  iterations = cms.uint32(300*1000))

process.c1 = cms.EDAnalyzer("ConsumingOneSharedResourceAnalyzer",
                            valueMustMatch = cms.untracked.int32(1),
                            moduleLabel = cms.untracked.InputTag("i1"),
                            resourceName = cms.untracked.string("foo"))
                            
process.c2 = cms.EDAnalyzer("ConsumingOneSharedResourceAnalyzer",
                            valueMustMatch = cms.untracked.int32(2),
                            moduleLabel = cms.untracked.InputTag("i2"),
                            resourceName = cms.untracked.string("foo"))

process.t = cms.Task(process.i1, process.i2)

process.p = cms.Path(process.c1+process.c2, process.t)

process.add_(cms.Service("ConcurrentModuleTimer",
                         modulesToExclude = cms.untracked.vstring("TriggerResults", "p"),
                         excludeSource = cms.untracked.bool(True)))

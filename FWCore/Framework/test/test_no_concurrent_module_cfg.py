import FWCore.ParameterSet.Config as cms

process = cms.Process("NOCONCURRENT")

process.source = cms.Source("EmptySource")

process.options = dict(
    numberOfStreams = 4,
    numberOfThreads = 4,
    numberOfConcurrentLuminosityBlocks = 1
)

process.maxEvents.input = 20

process.i1 = cms.EDProducer("BusyWaitIntOneSharedProducer", ivalue = cms.int32(1),
  iterations = cms.uint32(300*1000),
  resourceNames = cms.untracked.vstring("foo"))
process.i2 = cms.EDProducer("BusyWaitIntOneSharedProducer", ivalue = cms.int32(2),
  iterations = cms.uint32(300*1000),
  resourceNames = cms.untracked.vstring("foo"))
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

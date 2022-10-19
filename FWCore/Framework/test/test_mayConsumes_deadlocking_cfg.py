import FWCore.ParameterSet.Config as cms

process = cms.Process("DEADLOCKTEST")
process.source = cms.Source("EmptySource")

process.maxEvents.input = 20000

process.a = cms.EDAnalyzer("ConsumingOneSharedResourceAnalyzer", 
                            valueMustMatch = cms.untracked.int32(1),
                            moduleLabel = cms.untracked.InputTag("one"),
                            resourceName = cms.untracked.string("A")
                            )

process.b = cms.EDAnalyzer("ConsumingOneSharedResourceAnalyzer", 
                            valueMustMatch = cms.untracked.int32(2),
                            moduleLabel = cms.untracked.InputTag("two"),
                            resourceName = cms.untracked.string("B")
                            )

process.one = cms.EDProducer("IntOneSharedProducer",
                             ivalue = cms.int32(1),
                             resourceNames = cms.untracked.vstring("A", "B")
)

process.two = cms.EDProducer("IntOneSharedProducer",
                             ivalue = cms.int32(2),
                             resourceNames = cms.untracked.vstring("A", "B")
)
           
process.options = dict(
    numberOfThreads = 2,
    numberOfStreams = 0,
    numberOfConcurrentLuminosityBlocks = 1
)                 

process.t = cms.Task(process.one, process.two)

process.p1 = cms.Path(process.a, process.t)
process.p2 = cms.Path(process.b)

process.add_(cms.Service("ZombieKillerService", secondsBetweenChecks = cms.untracked.uint32(10)))

import FWCore.ParameterSet.Config as cms

process = cms.Process("DEADLOCKTEST")
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(20000))

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

process.one = cms.EDProducer("IntLegacyProducer",
                             ivalue = cms.int32(1)
)

process.two = cms.EDProducer("IntLegacyProducer",
                             ivalue = cms.int32(2)
)
           
process.options = cms.untracked.PSet(
                    allowUnscheduled = cms.untracked.bool(True),
                    numberOfThreads = cms.untracked.uint32(2),
                    numberOfStreams = cms.untracked.uint32(0)
)                 

process.p1 = cms.Path(process.a)
process.p2 = cms.Path(process.b)

process.add_(cms.Service("ZombieKillerService", secondsBetweenChecks = cms.untracked.uint32(10)))
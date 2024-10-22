import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")

process.WaitingService = cms.Service("WaitingService")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(2),
    numberOfStreams = cms.untracked.uint32(0)
)

process.waiter = cms.EDProducer("AcquireIntProducer",
                                streamsToAccumulate = cms.untracked.uint32(3),
                                tags = cms.VInputTag("busy1", "busy2"),
                                produceTag = cms.InputTag("busy3")
)

process.filterwaiter = cms.EDFilter("AcquireIntFilter",
                                    streamsToAccumulate = cms.untracked.uint32(3),
                                    tags = cms.VInputTag("busy1", "busy2"),
                                    produceTag = cms.InputTag("busy3")
)

process.streamwaiter = cms.EDProducer("AcquireIntStreamProducer",
                                      streamsToAccumulate = cms.untracked.uint32(3),
                                      tags = cms.VInputTag("busy1", "busy2"),
                                      produceTag = cms.InputTag("busy3")
)

process.streamfilterwaiter = cms.EDFilter("AcquireIntStreamFilter",
                                          streamsToAccumulate = cms.untracked.uint32(3),
                                          tags = cms.VInputTag("busy1", "busy2"),
                                          produceTag = cms.InputTag("busy3")
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))
process.busy2 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(2), iterations = cms.uint32(10*1000*1000))
process.busy3 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(2), iterations = cms.uint32(10*1000*1000))

process.tester = cms.EDAnalyzer("IntTestAnalyzer",
                                moduleLabel = cms.untracked.InputTag("waiter"),
                                valueMustMatch = cms.untracked.int32(5))

process.filtertester = cms.EDAnalyzer("IntTestAnalyzer",
                                      moduleLabel = cms.untracked.InputTag("filterwaiter"),
                                      valueMustMatch = cms.untracked.int32(5))

process.streamtester = cms.EDAnalyzer("IntTestAnalyzer",
                                      moduleLabel = cms.untracked.InputTag("streamwaiter"),
                                      valueMustMatch = cms.untracked.int32(5))

process.streamfiltertester = cms.EDAnalyzer("IntTestAnalyzer",
                                            moduleLabel = cms.untracked.InputTag("streamfilterwaiter"),
                                            valueMustMatch = cms.untracked.int32(5))

process.task = cms.Task(process.busy1, process.busy2, process.busy3,
                        process.waiter, process.filterwaiter,
                        process.streamwaiter, process.streamfilterwaiter)

process.testParentage1 = cms.EDAnalyzer("TestParentage",
                                        inputTag = cms.InputTag("waiter"),
                                        expectedAncestors = cms.vstring("busy1", "busy2", "busy3")
)

process.testParentage2 = cms.EDAnalyzer("TestParentage",
                                        inputTag = cms.InputTag("filterwaiter"),
                                        expectedAncestors = cms.vstring("busy1", "busy2", "busy3")
)

process.testParentage3 = cms.EDAnalyzer("TestParentage",
                                        inputTag = cms.InputTag("streamwaiter"),
                                        expectedAncestors = cms.vstring("busy1", "busy2", "busy3")
)

process.testParentage4 = cms.EDAnalyzer("TestParentage",
                                        inputTag = cms.InputTag("streamfilterwaiter"),
                                        expectedAncestors = cms.vstring("busy1", "busy2", "busy3")
)

process.p1 = cms.Path(process.tester + process.testParentage1, process.task)
process.p2 = cms.Path(process.filtertester + process.testParentage2, process.task)
process.p3 = cms.Path(process.streamtester + process.testParentage3, process.task)
process.p4 = cms.Path(process.streamfiltertester + process.testParentage4, process.task)

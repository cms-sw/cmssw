import FWCore.ParameterSet.Config as cms

process = cms.Process("TRACING")

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring("file:file_for_grapher.root"))

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(100*1000*1000))
process.shared1 = cms.EDProducer("BusyWaitIntOneSharedProducer",ivalue = cms.int32(12), iterations = cms.uint32(3000*1000))

process.adder1 = cms.EDProducer("AddIntsProducer",
                                labels = cms.VInputTag("a","busy1","shared1"))

process.busy2 = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(7), iterations = cms.uint32(4*1000*1000))
process.busy3 = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(2), iterations = cms.uint32(1000*1000))
process.shared2 = cms.EDProducer("BusyWaitIntOneSharedProducer", ivalue=cms.int32(-1), iterations = cms.uint32(6000*1000))

process.strt = cms.EDProducer("AddIntsProducer",
                              labels = cms.VInputTag("busy2","busy3","shared2","b", "adder1"))

process.t = cms.Task(process.busy1, process.shared1, process.adder1, process.busy2, process.busy3, process.shared2)

process.p = cms.Path(process.strt, process.t)

#exercise ES monitoring
process.testESSource = cms.ESSource("TestESConcurrentSource",
    firstValidLumis = cms.vuint32(1, 4, 6, 7, 8, 9),
    iterations = cms.uint32(10*1000*1000),
    checkIOVInitialization = cms.bool(True),
    expectedNumberOfConcurrentIOVs = cms.uint32(1)
)
process.concurrentIOVESProducer = cms.ESProducer("ConcurrentIOVESProducer")
process.test = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(False)
)
process.testOther = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(False),
                              fromSource = cms.untracked.ESInputTag(":other")
)
process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))
process.p1 = cms.Path(process.busy1 * process.test * process.testOther)

process.options = dict( numberOfStreams = 4,
                        numberOfThreads = 5,
                        numberOfConcurrentLuminosityBlocks = 1,
                        numberOfConcurrentRuns = 1
)

process.add_(cms.Service("Tracer", printTimestamps = cms.untracked.bool(True)))
process.add_(cms.Service("StallMonitor", fileName = cms.untracked.string("stallMonitor.log")))

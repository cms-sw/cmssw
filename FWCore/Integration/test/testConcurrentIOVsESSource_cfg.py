import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(8)
)

process.options = dict(
    numberOfThreads = 4,
    numberOfStreams = 4,
    numberOfConcurrentRuns = 1,
    numberOfConcurrentLuminosityBlocks = 4,
    eventSetup = dict(
        numberOfConcurrentIOVs = 2
    )
)

process.testESSource = cms.ESSource("TestESSource",
    firstValidLumis = cms.vuint32(1, 4, 6, 7, 8, 9),
    iterations = cms.uint32(10*1000*1000),
    checkIOVInitialization = cms.bool(True),
    expectedNumberOfConcurrentIOVs = cms.uint32(2)
)

process.concurrentIOVESProducer = cms.ESProducer("ConcurrentIOVESProducer")

process.test = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(False)
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.p1 = cms.Path(process.busy1 * process.test)

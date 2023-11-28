import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100)
)

process.maxEvents.input = 8

process.options = dict(
    numberOfThreads = 4,
    numberOfStreams = 4,
    numberOfConcurrentRuns = 1,
    numberOfConcurrentLuminosityBlocks = 4,
    eventSetup = dict(
        numberOfConcurrentIOVs = 2
    )
)

process.emptyESSourceI = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordI"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)

process.concurrentIOVESSource = cms.ESSource("ConcurrentIOVESSource",
    iovIsRunNotTime = cms.bool(True),
    firstValidLumis = cms.vuint32(1, 4, 6, 7, 8, 9),
    invalidLumis = cms.vuint32(),
    concurrentFinder = cms.bool(True)
)

process.concurrentIOVESProducer = cms.ESProducer("ConcurrentIOVESProducer")

process.acquireIntESProducer = cms.ESProducer("AcquireIntESProducer",
    numberOfIOVsToAccumulate = cms.untracked.uint32(2),
    secondsToWaitForWork = cms.untracked.uint32(1)
)

process.test = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(True),
                              # first 3 just ignored (indexed by cacheIdentifier that starts at 3 in this test)
                              # beginRun 1 + endRun 1 + beginLumi 1 + endLumi 3 + cacheIdentifier 3 = 9
                              # 9 + 5 (external work increments each of the 5 values by 1) = 14,
                              # runs are always 1, lumi ranges 1-3, 4-5, 6, 7, 8
                              # cacheIdentifier increments by 1 each time 3, 4, 5, 6, 7
                              expectedESAcquireTestResults = cms.untracked.vint32(0, 0, 0, 14, 20, 24, 27, 30),
                              expectedUniquePtrTestValue = cms.untracked.int32(102),
                              expectedOptionalTestValue = cms.untracked.int32(202)
)

process.emptyESSourceB = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordB"),
    firstValid = cms.vuint32(1),
    iovIsRunNotTime = cms.bool(True)
)
process.esTestAnalyzerB = cms.EDAnalyzer("ESTestAnalyzerB",
    runsToGetDataFor = cms.vint32(1),
    expectedValues = cms.untracked.vint32(11, 11, 11, 11, 11, 11, 11, 11)
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.p1 = cms.Path(process.busy1 * process.test * process.esTestAnalyzerB)

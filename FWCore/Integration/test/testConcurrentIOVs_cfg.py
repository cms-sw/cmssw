# This test should almost always be running 4 lumis concurrently
# and 2 IOVs concurrently. It prints out times that allow one
# to verify this by manually looking at the log file. We did
# not make the relationship between these times into a unit test
# pass/fail criteria because in unusual cases the relationship
# between times could vary. For example, if a thread got stuck on
# a very busy machine the times could be very different. We do not
# want unit tests that sometimes fail.

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

process.emptyESSourceI = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordI"),
    firstValid = cms.vuint32(1,100),
    iovIsRunNotTime = cms.bool(True)
)

process.emptyESSourceK = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordK"),
    firstValid = cms.vuint32(1,100),
    iovIsRunNotTime = cms.bool(True)
)

process.concurrentIOVESSource = cms.ESSource("ConcurrentIOVESSource",
    iovIsRunNotTime = cms.bool(True),
    firstValidLumis = cms.vuint32(1, 4, 6, 7, 8, 9),
    invalidLumis = cms.vuint32(),
    concurrentFinder = cms.bool(True)
)

process.concurrentIOVESProducer = cms.ESProducer("ConcurrentIOVESProducer")

process.test = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(True)
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.p1 = cms.Path(process.busy1 * process.test)

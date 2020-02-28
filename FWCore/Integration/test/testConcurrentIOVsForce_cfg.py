# Similar to testConcurrentIOVs
# Uses the forceNumberOfConcurrentIOVs to for
# 4 concurrent IOVs to be used for ESTestRecordA
# Should see 4 lumis and 4 IOVs running concurrently

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
        numberOfConcurrentIOVs = 4,
        forceNumberOfConcurrentIOVs = dict(
            ESTestRecordA = 4
        )
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
    concurrentFinder = cms.bool(True),
    testForceESSourceMode = cms.bool(True),
    findForRecordA = cms.bool(True)
)

process.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1)
)

process.concurrentIOVESProducer = cms.ESProducer("ConcurrentIOVESProducer")

process.test = cms.EDAnalyzer("ConcurrentIOVAnalyzer",
                              checkExpectedValues = cms.untracked.bool(True)
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.p1 = cms.Path(process.busy1 * process.test * process.esTestAnalyzerA)

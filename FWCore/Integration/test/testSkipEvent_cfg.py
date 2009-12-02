# Test the what happens after an exception associated
# with the behavior SkipEvent

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# We are testing the SkipEvent behavior.
process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('EventCorruption')
)

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.source = cms.Source("EmptySource",
    firstLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(100),
    firstEvent = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100)
)

process.testThrow = cms.EDAnalyzer("TestFailuresAnalyzer",
    whichFailure = cms.int32(5),
    eventToThrow = cms.untracked.uint32(2)
)

# In the path before the module throwing an exception all 3 events should run
process.beforeException = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 2,
        1, 1, 3,
        1, 1, 0,
        1, 0, 0
     )
)

# Note that this one checks that the second event was skipped
process.afterException = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 3,
        1, 1, 0,
        1, 0, 0
     )
)

process.onEndPath = cms.EDAnalyzer('RunLumiEventAnalyzer',
    verbose = cms.untracked.bool(True),
    expectedRunLumiEvents = cms.untracked.vuint32(
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 2,
        1, 1, 3,
        1, 1, 0,
        1, 0, 0
     ),
     dumpTriggerResults = cms.untracked.bool(True)
)

# The next two modules are not really necessary for the test
# Just adding in a producer and filter to make it more realistic
# No particular reason that I selected these two modules
process.thingWithMergeProducer = cms.EDProducer("ThingWithMergeProducer")

process.f1 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(98),
    onlyOne = cms.untracked.bool(False)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testSkipEvent.root'),
    SelectEvents = cms.untracked.PSet(
      SelectEvents = cms.vstring('p1')
    )
)

process.p1 = cms.Path(process.beforeException *
                      process.testThrow *
                      process.afterException *
                      process.thingWithMergeProducer *
                      process.f1)

process.p2 = cms.Path(process.afterException)

process.e = cms.EndPath(process.out * process.onEndPath)

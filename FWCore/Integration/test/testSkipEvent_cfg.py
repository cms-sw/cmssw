# Test the what happens after an exception associated
# with the behavior SkipEvent

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

# We are testing the SkipEvent behavior.
process.options.SkipEvent = 'EventCorruption'

process.maxEvents.input = 3

from FWCore.Modules.import EmptySource
process.source = EmptySource(
    firstLuminosityBlock = 1,
    numberEventsInLuminosityBlock = 100,
    firstEvent = 1,
    firstRun = 1,
    numberEventsInRun = 100
)

process.testThrow = cms.EDAnalyzer("TestFailuresAnalyzer",
    whichFailure = cms.int32(5),
    eventToThrow = cms.untracked.uint64(2)
)

from FWCore.Framework.modules import RunLumiEventAnalyzer, IntProducer, IntConsumingAnalyzer
# In the path before the module throwing an exception all 3 events should run
process.beforeException = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 2,
        1, 1, 3,
        1, 1, 0,
        1, 0, 0
     ]
)

# Note that this one checks that the second event was skipped
process.afterException = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 3,
        1, 1, 0,
        1, 0, 0
     ]
)

process.onEndPath = RunLumiEventAnalyzer(
    verbose = True,
    expectedRunLumiEvents = [
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
        1, 1, 2,
        1, 1, 3,
        1, 1, 0,
        1, 0, 0
     ],
     dumpTriggerResults = True
)

# The next two modules are not really necessary for the test
# Just adding in a producer and filter to make it more realistic
# No particular reason that I selected these two modules
from FWCore.Integration.modules import ThingWithMergeProducer
process.thingWithMergeProducer = ThingWithMergeProducer()

process.p1Done = IntProducer(ivalue = 1)
process.waitTillP1Done = IntConsumingAnalyzer(getFromModule = "p1Done")


process.f1 = cms.EDFilter("TestFilterModule",
    acceptValue = cms.untracked.int32(98),
    onlyOne = cms.untracked.bool(False)
)

from IOPool.Output.modules import PoolOutputModule
process.out = PoolOutputModule(
    fileName = 'testSkipEvent.root',
    SelectEvents = dict(SelectEvents = ['p1'])
)

process.p1 = cms.Path(process.beforeException *
                      process.testThrow *
                      process.afterException *
                      process.thingWithMergeProducer *
                      process.f1+process.p1Done)

process.p2 = cms.Path(process.waitTillP1Done+process.afterException)

process.e = cms.EndPath(process.out * process.onEndPath)

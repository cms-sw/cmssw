# The purpose of this unit test is to execute the code
# in the FastTimerService and at least verify that it
# runs without crashing. There is output in the log
# file and also a json file.

# The output times and memory usage values may vary from
# one job to the next. We do not want a test that can
# occasionally fail and require investigation when
# nothing is wrong, so we choose not to require certain
# values in the output.

# Besides the FastTimerService, the rest of the
# configuration is of no significance. This is
# just copied from another test configuration.
# We just needed some modules and paths to run
# in order to exercise the service.

import FWCore.ParameterSet.Config as cms

process = cms.Process("PROD1")

process.load('DQMServices.Core.DQMStore_cfi')

from HLTrigger.Timer.FastTimerService_cfi import *

process.FastTimerService = FastTimerService

process.FastTimerService.printEventSummary = True
process.FastTimerService.writeJSONSummary = True
process.FastTimerService.dqmTimeRange = 2000
process.FastTimerService.enableDQM = True
process.FastTimerService.enableDQMTransitions = True
process.FastTimerService.enableDQMbyLumiSection = True
process.FastTimerService.enableDQMbyModule = True
process.FastTimerService.enableDQMbyPath = True
process.FastTimerService.enableDQMbyProcesses = True

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        FastReport = cms.untracked.PSet(
            limit = cms.untracked.int32(1000000)
        ),
        threshold  = cms.untracked.string('INFO')
    )
)

process.options = cms.untracked.PSet(
    numberOfStreams = cms.untracked.uint32(1),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(1)
)

process.source = cms.Source("IntSource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(3)
)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testFastTimerService.root')
)

process.dqmOutput = cms.OutputModule('DQMRootOutputModule',
    fileName = cms.untracked.string('dqmOutput.root')
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.a1 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("source") ),
  expectedSum = cms.untracked.int32(530021),
  inputTagsNotFound = cms.untracked.VInputTag(
    cms.InputTag("source", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducer", processName=cms.InputTag.skipCurrentProcess()),
    cms.InputTag("intProducerU", processName=cms.InputTag.skipCurrentProcess())
  ),
  inputTagsBeginProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerBeginProcessBlock"),
  ),
  inputTagsEndProcessBlock = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock"),
  ),
  inputTagsEndProcessBlock2 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "two"),
  ),
  inputTagsEndProcessBlock3 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "three"),
  ),
  inputTagsEndProcessBlock4 = cms.untracked.VInputTag(
    cms.InputTag("intProducerEndProcessBlock", "four"),
  ),
  testGetterOfProducts = cms.untracked.bool(True)
)

process.a2 = cms.EDAnalyzer("TestFindProduct",
  inputTags = cms.untracked.VInputTag( cms.InputTag("intProducerA") ),
  expectedSum = cms.untracked.int32(300)
)

process.intProducer = cms.EDProducer("IntProducer", ivalue = cms.int32(1))

process.intProducerDeleted = cms.EDProducer("IntProducer", ivalue = cms.int32(10))

process.intProducerU = cms.EDProducer("IntProducer", ivalue = cms.int32(10))

process.intProducerA = cms.EDProducer("IntProducer", ivalue = cms.int32(100))

process.intVectorProducer = cms.EDProducer("IntVectorProducer",
  count = cms.int32(9),
  ivalue = cms.int32(11)
)

process.intProducerB = cms.EDProducer("IntProducer", ivalue = cms.int32(1000))

process.thingProducer = cms.EDProducer("ThingProducer")

process.intProducerBeginProcessBlock = cms.EDProducer("IntProducerBeginProcessBlock", ivalue = cms.int32(10000))

process.intProducerEndProcessBlock = cms.EDProducer("IntProducerEndProcessBlock", ivalue = cms.int32(100000))

process.t = cms.Task(process.intProducerDeleted,
                     process.intProducerU,
                     process.intProducerA,
                     process.intProducerB,
                     process.intVectorProducer,
                     process.intProducerBeginProcessBlock,
                     process.intProducerEndProcessBlock
)

process.p1 = cms.Path(process.intProducer * process.a1 * process.a2, process.t)

process.p2 = cms.Path(process.busy1 * process.thingProducer)

#process.e = cms.EndPath(process.out * process.dqmOutput)


import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(100)
)

# set to 6 output events. Actual number could be
# anywhere from 6 to 9 because after 6 events written
# it will finish the other events already running concurrently.
# In this config there are 3 other streams possible.
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(20),
  output = cms.untracked.int32(6)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(4),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(4)
)

process.busy1 = cms.EDProducer("BusyWaitIntProducer",
                               ivalue = cms.int32(1),
                               iterations = cms.uint32(10*1000*1000),
                               lumiNumberToThrow = cms.uint32(0)
)

process.p1 = cms.Path(process.busy1)
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testMaxEventsOutput.root')
)

process.e = cms.EndPath(process.out)


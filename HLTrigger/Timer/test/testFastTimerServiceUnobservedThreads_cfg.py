# Check that the FastTimerService handles a thread that it has not seen entering a TBB arena: the UnobservedThreadSignals
# test service emits the delayed-get signals of the "busy" module from a new thread after each event.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1

process.source = cms.Source("EmptySource")
process.maxEvents.input = 3

process.load("HLTrigger.Timer.FastTimerService_cfi")
process.FastTimerService.enableDQM = False
process.FastTimerService.printJobSummary = True

process.UnobservedThreadSignals = cms.Service("UnobservedThreadSignals",
    moduleLabel = cms.untracked.string("busy")
)

process.MessageLogger.cerr.FastReport = cms.untracked.PSet(limit = cms.untracked.int32(10000000))

process.busy = cms.EDProducer("BusyWaitIntProducer", ivalue = cms.int32(1), iterations = cms.uint32(1000*1000))

process.path = cms.Path(process.busy)

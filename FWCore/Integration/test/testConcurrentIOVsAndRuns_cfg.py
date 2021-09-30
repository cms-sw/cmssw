# Tests concurrent runs along with concurrent IOVs
# Note that 3 concurrent runs implies at least 7
# concurrent IOVs are needed and we configure
# 8 concurrent IOVs so that concurrent runs are
# really the limiting factor for the test.
# Note 7 includes 1 for the first run and then 3
# for each subsequent concurrent run which includes
# an IOV for end run, begin run, and begin lumi necessary
# to get to the next event. In this test every lumi is
# only valid for one transition (see internals of
# RunLumiESSource). This test checks that correct
# EventSetup info is retrieved in all the transitions
# plus the same test is run in a SubProcess to
# check that transitions there are also running properly.
# Manual examination of the times in the log output should
# show 3 events in 3 different runs being processed
# concurrently.

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1)
)

process.maxEvents.input = 30

process.options = dict(
    numberOfThreads = 8,
    numberOfStreams = 8,
    numberOfConcurrentRuns = 3,
    numberOfConcurrentLuminosityBlocks = 8,
    eventSetup = dict(
        numberOfConcurrentIOVs = 8
    )
)

process.runLumiESSource = cms.ESSource("RunLumiESSource")

process.test = cms.EDAnalyzer("RunLumiESAnalyzer")

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(40*1000*1000))

process.p1 = cms.Path(process.busy1 * process.test)

process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testConcurrentIOVsAndRuns.root')
)

process.e = cms.EndPath(process.out)

# ---------------------------------------------------------------

aSubProcess = cms.Process("TESTSUBPROCESS")
process.addSubProcess(cms.SubProcess(aSubProcess))

aSubProcess.runLumiESSource = cms.ESSource("RunLumiESSource")

aSubProcess.test = cms.EDAnalyzer("RunLumiESAnalyzer")

aSubProcess.p1 = cms.Path(aSubProcess.test)

aSubProcess.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('testConcurrentIOVsAndRunsSubProcess.root')
)

aSubProcess.e = cms.EndPath(aSubProcess.out)

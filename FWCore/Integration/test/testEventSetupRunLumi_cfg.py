import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(10)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30)
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(6),
    numberOfStreams = cms.untracked.uint32(6),
    numberOfConcurrentRuns = cms.untracked.uint32(1),
    numberOfConcurrentLuminosityBlocks = cms.untracked.uint32(6),
    eventSetup = cms.untracked.PSet(
        numberOfConcurrentIOVs = cms.untracked.uint32(5)
    )
)

process.runLumiESSource = cms.ESSource("RunLumiESSource")

process.test = cms.EDAnalyzer("RunLumiESAnalyzer")

process.busy1 = cms.EDProducer("BusyWaitIntProducer",ivalue = cms.int32(1), iterations = cms.uint32(10*1000*1000))

process.p1 = cms.Path(process.busy1 * process.test)

# ---------------------------------------------------------------

aSubProcess = cms.Process("TESTSUBPROCESS")
process.addSubProcess(cms.SubProcess(aSubProcess))

aSubProcess.runLumiESSource = cms.ESSource("RunLumiESSource")

aSubProcess.test = cms.EDAnalyzer("RunLumiESAnalyzer")

aSubProcess.p1 = cms.Path(aSubProcess.test)

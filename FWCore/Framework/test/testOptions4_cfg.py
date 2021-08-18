# Test of options when a looper is in the job
import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.options = dict(
    dumpOptions = True,
    numberOfThreads = 4,
    numberOfStreams = 4,
    numberOfConcurrentLuminosityBlocks = 7,
    eventSetup = dict(
        numberOfConcurrentIOVs = 7
    )
)

process.looper = cms.Looper("DummyLooper",
    value = cms.untracked.int32(4)
)

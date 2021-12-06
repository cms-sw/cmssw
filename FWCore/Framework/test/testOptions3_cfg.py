# Test of options when concurrentIOVs and concurrentLumis are too big
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
    numberOfConcurrentLuminosityBlocks = 5,
    eventSetup = dict(
        numberOfConcurrentIOVs = 5
    )
)

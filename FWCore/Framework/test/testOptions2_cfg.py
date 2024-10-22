# Test of options when parameters are zero
import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")
process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.options = dict(
    dumpOptions = True,
    numberOfThreads = 4,
    numberOfStreams = 0,
    numberOfConcurrentLuminosityBlocks = 0,
    eventSetup = dict(
        numberOfConcurrentIOVs = 0
    )
)

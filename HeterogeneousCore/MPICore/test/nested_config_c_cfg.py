import FWCore.ParameterSet.Config as cms

process = cms.Process("ConfigC")

process.options.numberOfThreads = 6
process.options.numberOfStreams = 6
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.load("FWCore.ParameterSet.MessageLogger")
process.MessageLogger.cerr.MPI = cms.untracked.PSet(
    reportEvery = cms.untracked.int32( 1 ),
    limit = cms.untracked.int32( 10000000 )
)

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource(
    mode = 'CommWorld',
    controllerProcessName = 'ConfigB'
)

process.maxEvents.input = -1

# Receive data from ConfigB
process.receiver = MPIReceiver(
    upstream = "source",
    instance = 99,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.finalcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("receiver")
)

process.path = cms.Path(process.receiver + process.finalcheck)
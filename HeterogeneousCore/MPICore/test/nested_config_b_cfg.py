import FWCore.ParameterSet.Config as cms

process = cms.Process("ConfigB")

process.options.numberOfThreads = 8
process.options.numberOfStreams = 8
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
    controllerProcessName = 'ConfigA'
)

process.maxEvents.input = -1

# Receive data from ConfigA
process.receiver = MPIReceiver(
    upstream = "source",
    instance = 42,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.checkA = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("receiver")
)

# Act as controller for ConfigC
process.mpiController = MPIController(
    mode = 'CommWorld',
    followerProcessName = 'ConfigC'
)

# Send (potentially modified) data to ConfigC
process.sender = MPISender(
    upstream = "mpiController",
    instance = 99,
    products = [ dict(
        type = "edm::EventID",
        name = 'receiver'
    )]
)

process.path = cms.Path(process.receiver + process.checkA + process.mpiController + process.sender)

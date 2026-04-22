import FWCore.ParameterSet.Config as cms

process = cms.Process("Follower")

process.options.numberOfThreads = 8
process.options.numberOfStreams = 8
process.options.numberOfConcurrentLuminosityBlocks = 2
process.options.numberOfConcurrentRuns = 2
process.options.wantSummary = False

process.load("FWCore.ParameterSet.MessageLogger")
process.MessageLogger.cerr.MPI = cms.untracked.PSet(
    reportEvery = cms.untracked.int32( 1 ),
    limit = cms.untracked.int32( 10000000 )
)

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource(
    mode = 'CommWorld',
    controllerProcessName = 'MPIController'
)

process.maxEvents.input = -1

# very verbose
#from HeterogeneousCore.MPICore.mpiReporter_cfi import mpiReporter as mpiReporter_
#process.reporter = mpiReporter_.clone()

process.receiver = MPIReceiver(
    upstream = "source",
    instance = 42,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.otherreceiver = MPIReceiver(
    upstream = "source",
    instance = 19,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.sender = MPISender(
    upstream = "otherreceiver", # guarantees that this module will only run after otherreceiver has run
    instance = 99,
    products = [ "edmEventID_otherreceiver__*" ]
)

process.analyzer = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("receiver")
)

process.path = cms.Path(process.receiver + process.analyzer + process.otherreceiver + process.sender)

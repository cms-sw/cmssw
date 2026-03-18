import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 8
process.options.numberOfStreams = 8
process.options.numberOfConcurrentLuminosityBlocks = 2
process.options.numberOfConcurrentRuns = 2
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource(
    mode = 'CommWorld',
    controller = 0
)

process.maxEvents.input = -1

# very verbose
#from HeterogeneousCore.MPICore.mpiReporter_cfi import mpiReporter as mpiReporter_
#process.reporter = mpiReporter_.clone()

process.receiver = MPIReceiver(
    upstream = "source",
    instance = 11,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.otherreceiver = MPIReceiver(
    upstream = "source",
    instance = 12,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.sender = MPISender(
    upstream = "otherreceiver", # guarantees that this module will only run after otherreceiver has run
    instance = 13,
    products = [ "edmEventID_otherreceiver__*" ]
)

process.analyzer = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("receiver")
)

process.path = cms.Path(process.receiver + process.analyzer + process.otherreceiver + process.sender)

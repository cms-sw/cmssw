import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIController")


# This test tests two different scenarios that can lead to errors.

# 1. It tests that streams that remain uninitialized are handled correctly by
# the MPI modules. This situation is reproduced by setting numberOfStreams
# higher than the number of events.

# 2. Due to the non-deterministic task scheduling in CMSSW, non-matching MPI
# modules might be scheduled in the local and remote process. This is ok, unless
# these modules contain blocking MPI calls. In this case, if they utilize all
# available working threads, the execution will deadlock. The probability of
# this to happen is close to 100% when numberOfStreams >> numberOfThreads

process.options.numberOfThreads = 4
process.options.numberOfStreams = 40
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from eventlist_cff import eventlist
process.source = cms.Source("EmptySourceFromEventIDs",
    events = cms.untracked(eventlist)
)

process.maxEvents.input = 10

from HeterogeneousCore.MPICore.modules import *

process.mpiController = MPIController(
    mode = 'CommWorld'
)

process.ids = cms.EDProducer("edmtest::EventIDProducer")

process.initialcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('ids')
)

process.sender = MPISender(
    upstream = "mpiController",
    instance = 42,
    products = [ "edmEventID_ids__*" ]
)

process.othersender = MPISender(
    upstream = "mpiController",
    instance = 19,
    products = [ "edmEventID_ids__*" ]
)

process.receiver = MPIReceiver(
    upstream = "othersender",   # guarantees that this module will only run after "othersender" has run
    instance = 99,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.finalcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('receiver')
)

process.path = cms.Path(process.mpiController + process.ids + process.initialcheck + process.sender + process.othersender + process.receiver + process.finalcheck)

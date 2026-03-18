import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIController")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

from eventlist_cff import eventlist
process.source = cms.Source("EmptySourceFromEventIDs",
    events = cms.untracked(eventlist)
)

process.maxEvents.input = 100

from HeterogeneousCore.MPICore.modules import *

process.ids = cms.EDProducer("edmtest::EventIDProducer")

process.initialcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('ids')
)

# Interface with the first remote process

process.mpiController1 = MPIController(
    mode = 'CommWorld',
    followers = [ 1, 2 ]
)

process.sender1 = MPISender(
    upstream = "mpiController1",
    instance = 11,
    products = [ "edmEventID_ids__*" ]
)

process.othersender1 = MPISender(
    upstream = "mpiController1",
    instance = 12,
    products = [ "edmEventID_ids__*" ]
)

process.receiver1 = MPIReceiver(
    upstream = "othersender1",   # guarantees that this module will only run after "othersender1" has run
    instance = 13,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.finalcheck1 = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('receiver1')
)

process.path1 = cms.Path(
    process.mpiController1 +
    process.ids +
    process.initialcheck +
    process.sender1 +
    process.othersender1 +
    process.receiver1 +
    process.finalcheck1
)

# Interface with the second remote process

process.mpiController2 = MPIController(
    mode = 'CommWorld',
    followers = [ 3, 4 ]
)

process.sender2 = MPISender(
    upstream = "mpiController2",
    instance = 21,
    products = [ "edmEventID_ids__*" ]
)

process.receiver2 = MPIReceiver(
    upstream = "sender2",   # guarantees that this module will only run after "sender2" has run
    instance = 22,
    products = [ dict(
        type = "edm::EventID",
        label = ""
    )]
)

process.finalcheck2 = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('receiver2')
)

process.path2 = cms.Path(
    process.mpiController2 +
    process.ids +
    process.initialcheck +
    process.sender2 +
    process.receiver2 +
    process.finalcheck2
)

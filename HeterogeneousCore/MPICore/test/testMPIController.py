import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIServer")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = 'file:server.uri'

from eventlist_cff import eventlist
process.source = cms.Source("EmptySourceFromEventIDs",
    events = cms.untracked(eventlist)
)

process.maxEvents.input = 100

from HeterogeneousCore.MPICore.mpiController_cfi import mpiController as mpiController_
process.mpiController = mpiController_.clone()

process.ids = cms.EDProducer("edmtest::EventIDProducer")

process.initialcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('ids')
)

process.sender = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(42),
    products = cms.vstring("edmEventID_ids__*")
)

process.othersender = cms.EDProducer("MPISender",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(19),
    products = cms.vstring("edmEventID_ids__*")
)

process.receiver = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("othersender"),  # guarantees that this module will only run after othersender has run
    instance = cms.int32(99),
    products = cms.VPSet(cms.PSet(
        type = cms.string("edm::EventID"),
        label = cms.string("")
    ))
)

process.finalcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('receiver')
)

process.path = cms.Path(process.mpiController + process.ids + process.initialcheck + process.sender + process.othersender + process.receiver + process.finalcheck)

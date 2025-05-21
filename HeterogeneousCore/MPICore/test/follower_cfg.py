import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIClient")

process.options.numberOfThreads = 8
process.options.numberOfStreams = 8
process.options.numberOfConcurrentLuminosityBlocks = 2
process.options.numberOfConcurrentRuns = 2
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.MPIService.pmix_server_uri = "file:server.uri"

process.source = cms.Source("MPISource")

process.maxEvents.input = -1

# very verbose
#from HeterogeneousCore.MPICore.mpiReporter_cfi import mpiReporter as mpiReporter_
#process.reporter = mpiReporter_.clone()

process.receiver = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("source"),
    instance = cms.int32(42),
    products = cms.VPSet(cms.PSet(
        type = cms.string("edm::EventID"),
        label = cms.string("")
    ))
)

process.otherreceiver = cms.EDProducer("MPIReceiver",
    upstream = cms.InputTag("source"),
    instance = cms.int32(19),
    products = cms.VPSet(cms.PSet(
        type = cms.string("edm::EventID"),
        label = cms.string("")
    ))
)

process.sender = cms.EDProducer("MPISender",
    upstream = cms.InputTag("otherreceiver"),  # guarantees that this module will only run after otherreceiver has run
    instance = cms.int32(99),
    products = cms.vstring("edmEventID_otherreceiver__*")
)

process.analyzer = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("receiver")
)

process.path = cms.Path(process.receiver + process.analyzer + process.otherreceiver + process.sender)

import FWCore.ParameterSet.Config as cms

process = cms.Process("ConfigA")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
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

from eventlist_cff import eventlist
process.source = cms.Source("EmptySourceFromEventIDs",
    events = cms.untracked(eventlist)
)

process.maxEvents.input = 100

from HeterogeneousCore.MPICore.modules import *

process.mpiController = MPIController(
    mode = 'CommWorld',
    followerProcessName = 'ConfigB'
)

process.ids = cms.EDProducer("edmtest::EventIDProducer")

process.initialcheck = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag('ids')
)

process.sender = MPISender(
    upstream = "mpiController",
    instance = 42,
    products = [ dict(
        type = "edm::EventID",
        name = 'ids'
    )]
)

process.path = cms.Path(process.mpiController + process.ids + process.initialcheck + process.sender)

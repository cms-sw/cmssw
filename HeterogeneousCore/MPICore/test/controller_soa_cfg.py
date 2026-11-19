import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIController")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
# MPIController supports a single concurrent LuminosityBlock
process.options.numberOfConcurrentLuminosityBlocks = 1
process.options.numberOfConcurrentRuns = 1
process.options.wantSummary = False

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

process.load("FWCore.ParameterSet.MessageLogger")
process.MessageLogger.cerr.MPI = cms.untracked.PSet(
    reportEvery = cms.untracked.int32( 1 ),
    limit = cms.untracked.int32( 10000000 )
)

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")

# produce and send a portable object, a portable collection, and some portable multicollections
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

from HeterogeneousCore.MPICore.modules import *

process.mpiController = MPIController(
    mode = 'CommWorld',
    followerProcessName = 'MPIFollower'
)

process.producePortableObjects = cms.EDProducer("TestAlpakaProducer@alpaka",
    size = cms.int32(42),
    size2 = cms.int32(33),
    size3 = cms.int32(61),
    alpaka = cms.untracked.PSet(
        # "serial_sync", "cuda_async", or "rocm_async"
        backend = cms.untracked.string("")
    )
)

process.sender = MPISender(
    upstream = "mpiController",
    instance = 42,
    products = [ dict(
        type = "PortableHostObject<portabletest::TestStruct>",
        name = 'producePortableObjects'
    ),
    dict(
        type = "PortableHostCollection<portabletest::TestSoALayout<128,false> >",
        name = 'producePortableObjects'
    ),
    dict(
        type = "PortableHostCollection<portabletest::SoABlocks2<128,false> >",
        name = 'producePortableObjects'
    ),
    dict(
        type = "PortableHostCollection<portabletest::SoABlocks3<128,false> >",
        name = 'producePortableObjects'
    ),
    dict(
        type = "ushort",
        name = 'producePortableObjects:backend'
    )]
)

# Same thing, but this time disabling TrivialSerialisation so all products are
# serialized through ROOT
process.senderNoTrivialSerialisation = MPISender(
    upstream = "sender",
    instance = 43,
    enableTrivialSerialisation = False,
    products = [ dict(
        type = "PortableHostObject<portabletest::TestStruct>",
        name = 'producePortableObjects'
    ),
    dict(
        type = "PortableHostCollection<portabletest::TestSoALayout<128,false> >",
        name = 'producePortableObjects'
    ),
    dict(
        type = "PortableHostCollection<portabletest::SoABlocks2<128,false> >",
        name = 'producePortableObjects'
    ),
    dict(
        type = "PortableHostCollection<portabletest::SoABlocks3<128,false> >",
        name = 'producePortableObjects'
    ),
    dict(
        type = "ushort",
        name = 'producePortableObjects:backend'
    )]
)

process.pathSoA = cms.Path(
    process.mpiController +
    process.producePortableObjects +
    process.sender +
    process.senderNoTrivialSerialisation
)

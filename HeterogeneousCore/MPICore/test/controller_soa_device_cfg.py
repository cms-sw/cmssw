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

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

# produce and send device collections
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

# a host product with no device counterpart, serialised through ROOT
process.ids = cms.EDProducer("edmtest::EventIDProducer")

process.sender = cms.EDProducer("MPISenderPortable@alpaka",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(42),
    products = cms.VPSet(
        # various device products
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceObject"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection2"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection3"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        # a host product with a portable trivial serialiser
        cms.PSet(
            type = cms.string("portabletest::TestHostCollection"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        # a host product with a non-portable trivial serialiser
        cms.PSet(
            type = cms.string("ushort"),
            src = cms.InputTag("producePortableObjects", "backend"),
        ),
        # a host product with no trivial serialiser
        cms.PSet(
            type = cms.string("edm::EventID"),
            src = cms.InputTag("ids", ""),
        ),
    )
)

process.pathSoA = cms.Path(
    process.mpiController +
    process.producePortableObjects +
    process.ids +
    process.sender
)

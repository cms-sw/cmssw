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

process.sender = cms.EDProducer("MPISenderPortable@alpaka",
    upstream = cms.InputTag("mpiController"),
    instance = cms.int32(42),
    products = cms.VPSet(
        cms.PSet(
            type = cms.string("portabletest::TestDeviceObject"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection2"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection3"),
            src = cms.InputTag("producePortableObjects", ""),
        ),
    )
)

process.pathSoA = cms.Path(
    process.mpiController +
    process.producePortableObjects +
    process.sender
)

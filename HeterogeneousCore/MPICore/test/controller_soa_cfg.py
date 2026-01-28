import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIServer")

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

# produce and send a portable object, a portable collection, and some portable multicollections
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

from HeterogeneousCore.MPICore.modules import *

process.mpiController = MPIController(
    mode = 'CommWorld'
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
    products = [
        "portabletestTestStructPortableHostObject_producePortableObjects__*",
        "128falseportabletestTestSoALayoutPortableHostCollection_producePortableObjects__*",
        "128falseportabletestSoABlocks2PortableHostCollection_producePortableObjects__*",
        "128falseportabletestSoABlocks3PortableHostCollection_producePortableObjects__*",
        "ushort_producePortableObjects_backend_*"
    ]
)

process.pathSoA = cms.Path(
    process.mpiController +
    process.producePortableObjects +
    process.sender
)

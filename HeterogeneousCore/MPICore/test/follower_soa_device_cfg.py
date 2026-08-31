import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")

# needed for MPIReceiverPortable and the automatic device-to-host conversion
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

from HeterogeneousCore.MPICore.modules import *

process.source = MPISource(mode = 'CommWorld',
    controllerProcessName = 'MPIController'
)

process.maxEvents.input = -1

# receive and validate a portable object, a portable collection, and some
# portable multi-block collections as device products
process.receiver = cms.EDProducer("MPIReceiverPortable@alpaka",
    upstream = cms.InputTag("source"),
    instance = cms.int32(42),
    products = cms.VPSet(
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceObject"),
            src = cms.InputTag("", ""),
        ),
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection"),
            src = cms.InputTag("", ""),
        ),
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection2"),
            src = cms.InputTag("", ""),
        ),
        cms.PSet(
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection3"),
            src = cms.InputTag("", ""),
        ),
    )
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("receiver")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("receiver")
)

process.pathSoA = cms.Path(
    process.receiver +
    process.validatePortableCollections +
    process.validatePortableObject
)

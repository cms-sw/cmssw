import FWCore.ParameterSet.Config as cms

process = cms.Process("MPIFollower")

process.options.numberOfThreads = 4
process.options.numberOfStreams = 4
process.options.wantSummary = False

process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")

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
        # a host product with a portable trivial serialiser
        cms.PSet(
            type = cms.string("portabletest::TestHostCollection"),
            src = cms.InputTag("hostPortable", ""),
        ),
        # a host product with a non-portable trivial serialiser
        cms.PSet(
            type = cms.string("ushort"),
            src = cms.InputTag("hostTrivial", ""),
        ),
        # a host product with no trivial serialiser, falling back to ROOT
        cms.PSet(
            type = cms.string("edm::EventID"),
            src = cms.InputTag("hostRoot", ""),
        ),
    )
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("receiver")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("receiver")
)

process.validateReceived = cms.EDAnalyzer("GenericConsumer",
    eventProducts = cms.untracked.vstring("receiver")
)

process.validateEventId = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("receiver", "hostRoot")
)

process.pathSoA = cms.Path(
    process.receiver +
    process.validatePortableCollections +
    process.validatePortableObject +
    process.validateReceived +
    process.validateEventId
)

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

# Produce, clone and validate a portable object, a portable collection, and some portable multicollections
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

# Produce portable objects on host (serial CPU backend)
process.producePortableObjects = cms.EDProducer("TestAlpakaProducer@alpaka",
    size = cms.int32(42),
    size2 = cms.int32(33),
    size3 = cms.int32(61),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("serial_sync")
    )
)

# a host product with no device counterpart, serialised through ROOT
process.ids = cms.EDProducer("edmtest::EventIDProducer")

# Alias "backend", to "backend2". This is to avoid "Duplicate Product
# Identifier" errors, because "backend" is already produced by
# clonePortableObjectsHtoH by default. "backend2" here has no meaning; is just
# something we use as an example of a host product with a non-portable trivial
# serialiser.
process.backendAlias = cms.EDAlias(
    producePortableObjects = cms.VPSet(
        cms.PSet(type = cms.string("*"), fromProductInstance =
                 cms.string("backend"), toProductInstance = cms.string("backend2"))
    )
)

# Clone from host to host, registering the H->D transformation
process.clonePortableObjectsHtoH = cms.EDProducer("ngt::GenericClonerPortable@alpaka",
    products = cms.VPSet(
        cms.PSet(
            src = cms.InputTag("producePortableObjects"),
            type = cms.string("portabletest::TestHostObject")
        ),
        cms.PSet(
            src = cms.InputTag("producePortableObjects"),
            type = cms.string("portabletest::TestHostCollection")
        ),
        cms.PSet(
            src = cms.InputTag("producePortableObjects"),
            type = cms.string("portabletest::TestHostCollection2")
        ),
        cms.PSet(
            src = cms.InputTag("producePortableObjects"),
            type = cms.string("portabletest::TestHostCollection3")
        ),
        # a host product with a non-portable trivial serialiser
        cms.PSet(
            src = cms.InputTag("backendAlias", "backend2"),
            type = cms.string("ushort")
        ),
        # a host product with no trivial serialiser, falling back to ROOT
        cms.PSet(
            src = cms.InputTag("ids"),
            type = cms.string("edm::EventID")
        ),
    ),
    verbose = cms.untracked.bool(True),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("")
    )
)

# Clone from device to device, registering the D->H transformation
process.clonePortableObjectsDtoD = cms.EDProducer("ngt::GenericClonerPortable@alpaka",
    products = cms.VPSet(
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceObject")
        ),
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection")
        ),
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection2")
        ),
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection3")
        ),
    ),
    verbose = cms.untracked.bool(True),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("")
    )
)

# Consume the products on host (via the D->H transformation registered above)
process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("clonePortableObjectsDtoD")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("clonePortableObjectsDtoD")
)

process.validateReceived = cms.EDAnalyzer("GenericConsumer",
    eventProducts = cms.untracked.vstring("clonePortableObjectsHtoH")
)

process.validateEventId = cms.EDAnalyzer("edmtest::EventIDValidator",
    source = cms.untracked.InputTag("clonePortableObjectsHtoH", "")
)

process.pathSoA = cms.Path(
    process.producePortableObjects +
    process.ids +
    process.clonePortableObjectsHtoH +
    process.clonePortableObjectsDtoD +
    process.validatePortableCollections +
    process.validatePortableObject +
    process.validateReceived +
    process.validateEventId
)

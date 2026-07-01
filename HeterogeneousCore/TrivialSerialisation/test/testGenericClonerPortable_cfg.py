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
            type = cms.string("portabletest::TestDeviceObject")
        ),
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("portabletest::TestDeviceCollection")
        ),
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("portabletest::TestDeviceCollection2")
        ),
        cms.PSet(
            src = cms.InputTag("clonePortableObjectsHtoH"),
            type = cms.string("portabletest::TestDeviceCollection3")
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

process.pathSoA = cms.Path(
    process.producePortableObjects +
    process.clonePortableObjectsHtoH +
    process.clonePortableObjectsDtoD +
    process.validatePortableCollections +
    process.validatePortableObject
)

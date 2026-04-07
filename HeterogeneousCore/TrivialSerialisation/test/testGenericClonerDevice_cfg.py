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

process.producePortableObjects = cms.EDProducer("TestAlpakaProducer@alpaka",
    size = cms.int32(42),
    size2 = cms.int32(33),
    size3 = cms.int32(61),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("")
    )
)

process.clonePortableObjects = cms.EDProducer("ngt::GenericClonerDevice@alpaka",
    eventProducts = cms.VPSet(
        cms.PSet(
            type = cms.string("portabletest::TestDeviceObject"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection2"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection3"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
    ),
    verbose = cms.untracked.bool(True),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("")
    )
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("clonePortableObjects")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("clonePortableObjects")
)


# TODO: Automatic Device <-> Host conversions are currently not supported when
# the types are not known at compile-time.
process.pathSoA = cms.Path(
    process.producePortableObjects +
    process.clonePortableObjects
    # process.validatePortableCollections +
    # process.validatePortableObject
)

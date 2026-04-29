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

process.clonePortableObjectsOnHost = cms.EDProducer("ngt::GenericClonerHost@alpaka",
    eventProducts = cms.VPSet(
        cms.PSet(
            type = cms.string("portabletest::TestHostObject"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestHostCollection"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestHostCollection2"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestHostCollection3"),
            label = cms.string("producePortableObjects"),
            instance = cms.string("")
        ),
    ),
    verbose = cms.untracked.bool(True),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("")
    )
)

process.clonePortableObjectsOnDevice = cms.EDProducer("ngt::GenericClonerDevice@alpaka",
    eventProducts = cms.VPSet(
        cms.PSet(
            type = cms.string("portabletest::TestDeviceObject"),
            label = cms.string("clonePortableObjectsOnHost"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection"),
            label = cms.string("clonePortableObjectsOnHost"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection2"),
            label = cms.string("clonePortableObjectsOnHost"),
            instance = cms.string("")
        ),
        cms.PSet(
            type = cms.string("portabletest::TestDeviceCollection3"),
            label = cms.string("clonePortableObjectsOnHost"),
            instance = cms.string("")
        ),
    ),
    verbose = cms.untracked.bool(True),
    alpaka = cms.untracked.PSet(
        backend = cms.untracked.string("")
    )
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("clonePortableObjectsOnDevice")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("clonePortableObjectsOnDevice")
)

process.pathSoA = cms.Path(
    process.producePortableObjects +
    process.clonePortableObjectsOnHost +
    process.clonePortableObjectsOnDevice +
    process.validatePortableCollections +
    process.validatePortableObject
)

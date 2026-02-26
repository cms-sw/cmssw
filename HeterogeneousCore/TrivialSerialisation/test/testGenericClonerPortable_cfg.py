import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.INFO.limit = 10000000

process.options.numberOfThreads = 1
process.options.numberOfStreams = 1

process.source = cms.Source("EmptySource")
process.maxEvents.input = 10

#produce, clone and validate a portable object, a portable collection, and some portable multicollections
process.load("Configuration.StandardSequences.Accelerators_cff")
process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")

process.producePortableObjects = cms.EDProducer("TestAlpakaProducer@alpaka",
    size = cms.int32(42),
    size2 = cms.int32(33),
    size3 = cms.int32(61),
    alpaka = cms.untracked.PSet(
        # "serial_sync", "cuda_async", or "rocm_async"
        backend = cms.untracked.string("")
    )
)

process.clonePortableObjects = cms.EDProducer("GenericClonerPortable@alpaka",
    eventProducts = cms.vstring("producePortableObjects"),
    verbose = cms.untracked.bool(True),
    alpaka = cms.untracked.PSet(
        # "serial_sync", "cuda_async", or "rocm_async"
        backend = cms.untracked.string("")
    )
)

process.validatePortableCollections = cms.EDAnalyzer("TestAlpakaAnalyzer",
    source = cms.InputTag("clonePortableObjects")
)

process.validatePortableObject = cms.EDAnalyzer("TestAlpakaObjectAnalyzer",
    source = cms.InputTag("clonePortableObjects")
)

process.taskSoA = cms.Task(
    process.producePortableObjects,
    process.clonePortableObjects
)

process.pathSoA = cms.Path(
    process.validatePortableCollections +
    process.validatePortableObject,
    process.taskSoA
)

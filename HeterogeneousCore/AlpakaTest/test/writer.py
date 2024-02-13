import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import *

process = cms.Process('Writer')

process.source = cms.Source('EmptySource')

process.load('Configuration.StandardSequences.Accelerators_cff')
process.load('HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi')

# enable logging for the analysers
process.MessageLogger.TestAlpakaAnalyzer = cms.untracked.PSet()
process.MessageLogger.TestAlpakaObjectAnalyzer = cms.untracked.PSet()

# either run the producer on a gpu (if available) and copy the product to the cpu, or run the producer directly on the cpu
process.testProducer = cms.EDProducer('TestAlpakaProducer@alpaka',
    size = cms.int32(42),
    size2 = cms.int32(33),
    size3 = cms.int32(61)
    # alpaka.backend can be set to a specific backend to force using it, or be omitted or left empty to use the defult backend;
    # depending on the architecture and available hardware, the supported backends are "serial_sync", "cuda_async", "rocm_async"
    #alpaka = cms.untracked.PSet(
    #    backend = cms.untracked.string("")
    #)
)

# analyse the first set of products
process.testAnalyzer = cms.EDAnalyzer('TestAlpakaAnalyzer',
    source = cms.InputTag('testProducer')
)

process.testObjectAnalyzer = cms.EDAnalyzer('TestAlpakaObjectAnalyzer',
    source = cms.InputTag('testProducer')
)

# run a second producer explicitly on the cpu
process.testProducerSerial = makeSerialClone(process.testProducer,
    size = cms.int32(99),
    size2 = cms.int32(51),
    size3 = cms.int32(43)
)

# analyse the second set of products
process.testAnalyzerSerial = cms.EDAnalyzer('TestAlpakaAnalyzer',
    source = cms.InputTag('testProducerSerial'),
    expectBackend = cms.string('SerialSync')
)

process.testObjectAnalyzerSerial = cms.EDAnalyzer('TestAlpakaObjectAnalyzer',
    source = cms.InputTag('testProducerSerial'),
    expectBackend = cms.string('SerialSync')
)

# write all products to a 'test.root' file
process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

process.process_path = cms.Path(
    process.testProducer +
    process.testAnalyzer +
    process.testObjectAnalyzer)

process.serial_path = cms.Path(
    process.testProducerSerial +
    process.testAnalyzerSerial +
    process.testObjectAnalyzerSerial)

process.output_path = cms.EndPath(process.output)

process.maxEvents.input = 10

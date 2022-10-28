import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

process = cms.Process('Writer')

process.source = cms.Source('EmptySource')

process.load('Configuration.StandardSequences.Accelerators_cff')

# enable logging for the TestPortableAnalyzer
process.MessageLogger.TestPortableAnalyzer = cms.untracked.PSet()

# run the producer on a CUDA gpu (if available)
process.testProducerCuda = cms.EDProducer('TestPortableProducerCUDA',
    size = cms.int32(42)
)

# copy the product from the gpu (if available) to the host
process.testTranscriberFromCuda = cms.EDProducer('TestPortableTranscriber',
    source = cms.InputTag('testProducerCuda')
)

# run the producer on the cpu
process.testProducerCpu = cms.EDProducer('TestPortableProducerCPU',
    size = cms.int32(42)
)

# either run the producer on a CUDA gpu (if available) and copy the product to the cpu, or run the producer directly on the cpu
process.testProducer = SwitchProducerCUDA(
    cpu = cms.EDAlias(
        testProducerCpu = cms.VPSet(cms.PSet(type = cms.string('*')))
    ),
    cuda = cms.EDAlias(
        testTranscriberFromCuda = cms.VPSet(cms.PSet(type = cms.string('*')))
    )
)

# analyse the product
process.testAnalyzer = cms.EDAnalyzer('TestPortableAnalyzer',
    source = cms.InputTag('testProducer')
)

# run a second producer explicitly on the cpu
process.testProducerSerial = cms.EDProducer('TestPortableProducerCPU',
    size = cms.int32(99)
)

# analyse the second product
process.testAnalyzerSerial = cms.EDAnalyzer('TestPortableAnalyzer',
    source = cms.InputTag('testProducerSerial')
)

# write the two products to a 'test.root' file
process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_testProducer_*_*',
        'keep *_testProducerSerial_*_*',
  )
)

process.producer_task = cms.Task(process.testProducerCuda, process.testTranscriberFromCuda, process.testProducerCpu)

process.process_path = cms.Path(
    process.testProducer +
    process.testAnalyzer,
    process.producer_task)

process.serial_path = cms.Path(
    process.testProducerSerial +
    process.testAnalyzerSerial)

process.output_path = cms.EndPath(process.output)

process.maxEvents.input = 10

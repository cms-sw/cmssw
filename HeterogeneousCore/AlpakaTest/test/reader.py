import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')

# read the products from a 'test.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:test.root')
)

# enable logging for the analysers
process.MessageLogger.TestAlpakaAnalyzer = cms.untracked.PSet()
process.MessageLogger.TestAlpakaObjectAnalyzer = cms.untracked.PSet()

# analyse the first set of products
process.testAnalyzer = cms.EDAnalyzer('TestAlpakaAnalyzer',
    source = cms.InputTag('testProducer')
)

process.testObjectAnalyzer = cms.EDAnalyzer('TestAlpakaObjectAnalyzer',
    source = cms.InputTag('testProducer')
)

# analyse the second product
process.testAnalyzerSerial = cms.EDAnalyzer('TestAlpakaAnalyzer',
    source = cms.InputTag('testProducerSerial'),
    expectBackend = cms.string('SerialSync')
)

process.testObjectAnalyzerSerial = cms.EDAnalyzer('TestAlpakaObjectAnalyzer',
    source = cms.InputTag('testProducerSerial'),
    expectBackend = cms.string('SerialSync')
)

process.device_path = cms.Path(process.testAnalyzer + process.testObjectAnalyzer)

process.serial_path = cms.Path(process.testAnalyzerSerial + process.testObjectAnalyzerSerial)

process.maxEvents.input = 10

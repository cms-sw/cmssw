import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')

# read the products from a 'test.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:test.root')
)

# enable logging for the TestAlpakaAnalyzer
process.MessageLogger.TestAlpakaAnalyzer = cms.untracked.PSet()

# analyse the first product
process.testAnalyzer = cms.EDAnalyzer('TestAlpakaAnalyzer',
    source = cms.InputTag('testProducer')
)

# analyse the second product
process.testAnalyzerSerial = cms.EDAnalyzer('TestAlpakaAnalyzer',
    source = cms.InputTag('testProducerSerial')
)

process.cuda_path = cms.Path(process.testAnalyzer)

process.serial_path = cms.Path(process.testAnalyzerSerial)

process.maxEvents.input = 10

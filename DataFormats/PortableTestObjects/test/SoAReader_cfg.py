import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# read the products from a 'test.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:test.root')
)

# enable logging for the analyser
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

process.soAAnalyzer = cms.EDAnalyzer('SchemaEvolutionAnalyzer',
    source = cms.InputTag("soaproducer", "SchemaEvolutionProduct"),
)

process.p = cms.Path(process.soAAnalyzer)

import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# read the products from a 'test.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/maholzer/SchemaEvolutionTestData/SEFour.root')
)

# enable logging for the analyser
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

process.evolutionFourAnalyzer = cms.EDAnalyzer('EvolutionFourAnalyzer',
    source = cms.InputTag("soaproducer", "EvolutionFourProduct"),
)

process.p = cms.Path(process.evolutionFourAnalyzer)

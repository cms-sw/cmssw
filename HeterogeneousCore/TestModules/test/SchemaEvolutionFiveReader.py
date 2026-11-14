import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# read the products from a 'test.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/maholzer/SchemaEvolutionTestData/SEFive.root')
)

# enable logging for the analyser
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

process.evolutionFiveAnalyzer = cms.EDAnalyzer('EvolutionFiveAnalyzer',
    source = cms.InputTag("soaproducer", "EvolutionFiveProduct"),
)

process.p = cms.Path(process.evolutionFiveAnalyzer)

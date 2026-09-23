import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# read the products from a 'testTwo.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/maholzer/SchemaEvolutionTestData/SETwo.root')
)

# enable logging for the analyser
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

process.evolutionTwoAnalyzer = cms.EDAnalyzer('EvolutionTwoAnalyzer',
    source = cms.InputTag("soaproducer", "EvolutionTwoProduct"),
)

process.p = cms.Path(process.evolutionTwoAnalyzer)

import FWCore.ParameterSet.Config as cms

process = cms.Process('Reader')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

# read the products from a 'testThree.root' file
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/user/m/maholzer/SchemaEvolutionTestData/SEThree.root')
)

# enable logging for the analyser
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

process.evolutionThreeAnalyzer = cms.EDAnalyzer('EvolutionThreeAnalyzer',
    source = cms.InputTag("soaproducer", "EvolutionThreeProduct"),
)

process.p = cms.Path(process.evolutionThreeAnalyzer)

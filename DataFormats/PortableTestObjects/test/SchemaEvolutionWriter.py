import FWCore.ParameterSet.Config as cms

process = cms.Process("Writer")
process.source = cms.Source('EmptySource')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# enable logging for the analysers
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

# Produce zeroth evolution of the SoA product
process.soaproducer = cms.EDProducer("SchemaEvolutionSoAProducer")

# write all products to a 'test.root' file
process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('SchemaEvolutionZero.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

# Add to process path
process.p = cms.Path(process.soaproducer)
process.output_path = cms.EndPath(process.output)

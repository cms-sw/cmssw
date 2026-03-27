import FWCore.ParameterSet.Config as cms

process = cms.Process("Writer")
process.source = cms.Source('EmptySource')
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(1))

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# enable logging for the analysers
process.MessageLogger.soAAnalyzer = cms.untracked.PSet()

# Produce an SoA and add it to the event
process.soaproducer = cms.EDProducer("SchemaEvolutionProducer")

# write all products to a 'test.root' file
process.output = cms.OutputModule('PoolOutputModule',
    fileName = cms.untracked.string('test.root'),
    outputCommands = cms.untracked.vstring('keep *')
)

# Add to process path
process.p = cms.Path(process.soaproducer)
process.output_path = cms.EndPath(process.output)

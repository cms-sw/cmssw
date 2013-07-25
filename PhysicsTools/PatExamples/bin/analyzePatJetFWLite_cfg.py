import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.FWLiteParams = cms.PSet(
    inputFile   = cms.string('file:jet2011A_aod.root'),
    outputFile  = cms.string('analyzePatBasics.root'),
    jets = cms.InputTag('goodPatJets')
)


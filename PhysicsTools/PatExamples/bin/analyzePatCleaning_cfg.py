import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.FWLiteParams = cms.PSet(
    inputFile    = cms.string('file:cocTuple.root'),
    outputFile   = cms.string('analyzePatCleaning.root'),
    jets     = cms.InputTag('cocPatJets'),
    overlaps = cms.string('electrons')
)

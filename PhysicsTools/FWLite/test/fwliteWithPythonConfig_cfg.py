import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.MuonAnalyzer = cms.PSet(
    ## common input for wrapped analyzers
    fileNames   = cms.vstring('file:patTuple.root'),  ## mandatory
    outputFile  = cms.string('analyzeFWLiteHistograms.root'),## mandatory
    maxEvents   = cms.int32(-1),                      ## optional
    outputEvery = cms.uint32(10),                     ## optional
    ## input specific for this analyzer
    muons = cms.InputTag('cleanPatMuons')
)

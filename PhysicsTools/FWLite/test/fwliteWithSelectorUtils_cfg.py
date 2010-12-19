import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.WBosonAnalyzer = cms.PSet(
    ## common input for wrapped analyzers
    fileNames   = cms.vstring('file:patTuple.root'),  ## mandatory
    outputFile  = cms.string('analyzeFWLiteHistograms.root'),## mandatory
    maxEvents   = cms.int32(-1),                      ## optional
    outputEvery = cms.uint32(10),                     ## optional
    ## parameters for W boson selector
    selection = cms.PSet(
        muonSrc      = cms.InputTag('cleanPatMuons'),
        metSrc       = cms.InputTag('patMETs'),
        muonPtMin    = cms.double(20.0),
        metMin       = cms.double(20.0),
        cutsToIgnore = cms.vstring('MET')
        )
)


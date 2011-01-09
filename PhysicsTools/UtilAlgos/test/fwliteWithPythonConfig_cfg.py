import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

##process.fwliteParameters = cms.PSet(
process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('file:patTuple.root'),         ## mandatory
    maxEvents   = cms.int32(-1),                             ## optional
    outputEvery = cms.uint32(10),                            ## optional
)
    
process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzeFWLiteHistograms.root'),## mandatory
)

process.muonAnalyzer = cms.PSet(
    ## input specific for this analyzer
    muons = cms.InputTag('cleanPatMuons')
)

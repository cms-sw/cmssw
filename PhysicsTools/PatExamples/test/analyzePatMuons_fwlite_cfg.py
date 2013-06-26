import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('file:patTuple.root'),         ## mandatory
    maxEvents   = cms.int32(100),                            ## optional
    outputEvery = cms.uint32(10),                            ## optional
)
    
process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzePatMuons.root'),          ## mandatory
)

process.patMuonAnalyzer = cms.PSet(
    ## input specific for this analyzer
    muons = cms.InputTag('cleanPatMuons')
)

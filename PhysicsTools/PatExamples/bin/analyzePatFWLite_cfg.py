import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring(['file:patTuple.root']),  ## mandatory
    maxEvents   = cms.int32(-1),
    outputEvery = cms.uint32(10)
    )

process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzePatBasics.root') ## mandatory
    )

process.MuonAnalyzer = cms.PSet(
    muons = cms.InputTag('cleanPatMuons')
)


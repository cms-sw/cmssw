import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring(['file:patTuple.root']),  ## mandatory
    maxEvents   = cms.int32(-1),
    outputEvery = cms.uint32(10)
    )

process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzePatBasics_fwlite.root') ## mandatory
    )

process.muonAnalyzer = cms.PSet(
    muons = cms.InputTag('cleanPatMuons')
)


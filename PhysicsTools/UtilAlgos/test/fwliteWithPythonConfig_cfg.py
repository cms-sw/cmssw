import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('root://eoscms//eos/cms/store/relval/CMSSW_5_3_6-START53_V14/RelValZTT/GEN-SIM-RECO/v2/00000/4E4AD1B8-FC29-E211-B998-001A928116B4.root'), ## mandatory
    maxEvents   = cms.int32(100),                            ## optional
    outputEvery = cms.uint32(10),                            ## optional
)

process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzeFWLiteHistograms.root'),  ## mandatory
)

process.muonAnalyzer = cms.PSet(
    ## input specific for this analyzer
    muons = cms.InputTag('muons')
)

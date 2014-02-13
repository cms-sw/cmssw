import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('root://eoscms//eos/cms/store/relval/CMSSW_7_0_0_pre11/RelValProdTTbar/GEN-SIM-RECO/START70_V4-v1/00000/0EA82C3C-646A-E311-9CB3-0025905A6070.root'),                                ## mandatory
    maxEvents   = cms.int32(-1),                             ## optional
    outputEvery = cms.uint32(10),                            ## optional
)

process.fwliteOutput = cms.PSet(
    fileName  = cms.string('analyzeFWLiteHistograms.root'),  ## mandatory
)

process.selection = cms.PSet(
        muonSrc      = cms.InputTag('muons'),
        metSrc       = cms.InputTag('metJESCorAK5CaloJetMuons'),
        muonPtMin    = cms.double(20.0),
        metMin       = cms.double(20.0),
       #cutsToIgnore = cms.vstring('MET')
)


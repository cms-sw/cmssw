import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('root://eoscms//eos/cms/store/relval/CMSSW_7_1_0_pre1/RelValProdTTbar/GEN-SIM-RECO/START70_V5-v1/00000/14842A6B-2086-E311-B5CB-02163E00E8DA.root'),                                ## mandatory
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


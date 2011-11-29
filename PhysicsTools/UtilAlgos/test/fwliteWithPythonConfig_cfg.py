import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames   = cms.vstring('root://eoscms//eos/cms/store/relval/CMSSW_5_0_0_pre6/RelValProdTTbar/GEN-SIM-RECO/START50_V5-v1/0196/F2D6A76C-9C16-E111-A52E-003048D3FC94.root?svcClass=default'), ## mandatory
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

import FWCore.ParameterSet.Config as cms

process = cms.PSet()

process.fwliteInput = cms.PSet(
     fileNames   = cms.vstring('root://eoscms.cern.ch//eos/cms/store/user/cmsbuild/store/relval/CMSSW_9_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU25ns_92X_upgrade2017_realistic_v7-v1/00000/32EA1438-3D61-E711-8FE7-0025905B85B2.root'), ## mandatory
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

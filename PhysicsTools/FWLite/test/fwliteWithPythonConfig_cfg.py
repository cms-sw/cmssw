import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

process.MuonAnalyzer = cms.PSet(
    ## common input for wrapped analyzers
    fileNames   = cms.vstring('rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_8_6/RelValTTbar/GEN-SIM-RECO/START38_V13-v1/0065/2856A4C7-B5E7-DF11-BE1D-00304867BFA8.root'),  ## mandatory
    outputFile  = cms.string('analyzeFWLiteHistograms.root'),## mandatory
    maxEvents   = cms.int32(-1),                      ## optional
    reportAfter = cms.uint32(10),                     ## optional
    ## input specific for this analyzer
    muons = cms.InputTag('muons')
)

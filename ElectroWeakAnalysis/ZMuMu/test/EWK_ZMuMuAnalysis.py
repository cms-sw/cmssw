import FWCore.ParameterSet.Config as cms

process = cms.Process("TestZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.options.SkipEvent = cms.untracked.vstring('ProductNotFound')
process.options.FailPath = cms.untracked.vstring('ProductNotFound')


process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
#    'file:/scratch1/cms/data/summer09/aodsim/zmumu/0016/889E7356-0084-DE11-AF48-001E682F8676.root'
#    'file:testEWKMuSkim.root'
  "rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/F8EE38AF-1EBE-DE11-8D19-00304891F14E.root"
    
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START3X_V21::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

### subskim
from ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff import *


process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)

### analysis
from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuCategories_oneshot.root")
)


### vertexing
#process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")

### plots
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesNtuples_cff") 
process.ntuplesOut.fileName = cms.untracked.string('NtupleLoose_test_oneshot.root')



# SubSkim Output module configuration

process.zMuMuSubskimOutputModule.fileName = 'testZMuMuSubskim_oneshot.root'


process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisSchedules_cff") 


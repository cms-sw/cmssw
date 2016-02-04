import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.options.SkipEvent = cms.untracked.vstring('ProductNotFound')
process.options.FailPath = cms.untracked.vstring('ProductNotFound')


process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100


# Input files
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0041/1C1BBE0B-D2D2-DF11-BDA3-002618943852.root'
    )
)
#import os
#dirname = "/tmp/degrutto/MinBiasMC/"
#dirlist = os.listdir(dirname)
#basenamelist = os.listdir(dirname + "/")
#for basename in basenamelist:
#                process.source.fileNames.append("file:" + dirname + "/" + basename)
#                print "Number of files to process is %s" % (len(process.source.fileNames))

                


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V12::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

### Subskim

############
## to run on data or without MC truth uncomment the following
#process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff")
############

# output module configuration
process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")

############
## to run the MC truth uncomment the following
## Look also at python/ZMuMuAnalysisSchedules_cff.py
process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff")
process.zMuMuSubskimOutputModule.outputCommands.extend(process.mcEventContent.outputCommands)
####

process.zMuMuSubskimOutputModule.fileName = 'file:/tmp/fabozzi/testZMuMuSubskim_oneshot_Test.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)

### Here set the HLT Path for trigger matching
process.muonTriggerMatchHLTMuons.pathNames = cms.vstring( 'HLT_Mu11' )
process.userDataMuons.hltPath = cms.string("HLT_Mu11")
process.userDataDimuons.hltPath = cms.string("HLT_Mu11")
process.userDataDimuonsOneTrack.hltPath = cms.string("HLT_Mu11")
############

### Analysis
from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('ewkZMuMuCategories_oneshot_Test.root')
)


### vertexing
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")
process.vtxedNtuplesOut.fileName = cms.untracked.string('file:/tmp/fabozzi/VtxedNtupleLoose_test.root')

### 3_5_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.patTrigger.processName = "REDIGI"
#process.patTriggerEvent.processName = "REDIGI"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI" )

### 3_6_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","REDIGI36X")
#process.patTrigger.processName = "REDIGI36X"
#process.patTriggerEvent.processName = "REDIGI36X"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI36X" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI36X" )

### plots
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisNtupler_cff")
process.ntuplesOut.fileName = cms.untracked.string('file:/tmp/fabozzi/NtupleLooseTestNew_oneshot_all_Test.root')

###
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisSchedules_cff") 


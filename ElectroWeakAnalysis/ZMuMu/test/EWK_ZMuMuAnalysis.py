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
    'file:/scratch2/users/fabozzi/spring10/zmm/38262142-DF46-DF11-8238-0030487C6A90.root'
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
process.GlobalTag.globaltag = cms.string('START36_V8::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

### subskim
## to run on data uncomment the following
#process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff")

# output module
process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")

## to run the MC truth uncomment the following
process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff")
process.zMuMuSubskimOutputModule.outputCommands.extend(process.mcEventContent.outputCommands)
####

process.zMuMuSubskimOutputModule.fileName = 'file:testZMuMuSubskim_oneshot_Test.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)

### analysis
from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string('ewkZMuMuCategories_oneshot_Test.root')
)


### vertexing
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")
process.vtxedNtuplesOut.fileName = cms.untracked.string('file:VtxedNtupleLoose_test.root')

### to process REDIGI HLT tables uncomment the following
process.patTrigger.processName = "REDIGI"
process.patTriggerEvent.processName = "REDIGI"
process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI" )
process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI" )

### plots
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisNtupler_cff")
process.ntuplesOut.fileName = cms.untracked.string('file:NtupleLooseTestNew_oneshot_all_Test.root')

###
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisSchedules_cff") 


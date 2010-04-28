

import FWCore.ParameterSet.Config as cms

process = cms.Process("ZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.options.SkipEvent = cms.untracked.vstring('ProductNotFound')
process.options.FailPath = cms.untracked.vstring('ProductNotFound')


process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 100


# Input files (on disk)
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
'rfio:/castor/cern.ch/user/f/fabozzi/mc7tev/F8EE38AF-1EBE-DE11-8D19-00304891F14E.root'

   
    )
                            )
#import os
#dirname = "/tmp/degrutto/MinBiasMC/"
#dirlist = os.listdir(dirname)
#basenamelist = os.listdir(dirname + "/")
#for basename in basenamelist:
#                process.source.fileNames.append("file:" + dirname + "/" + basename)
#                print "Number of files to process is %s" % (len(process.source.fileNames))

                


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START3X_V21::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

### subskim
##from ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff import *

process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")




process.zMuMuSubskimOutputModule.fileName = 'testZMuMuSubskim_oneshot_Test.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)





### analysis
from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuCategories_oneshot_all_3_Test.root")
)


### vertexing
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesVtxed_cff")

### plots
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesPlots_cff")

### ntuple
process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisNtupler_cff")
process.ntuplesOut.fileName = cms.untracked.string('file:NtupleLooseTestNew_oneshot_all_Test.root')



# SubSkim Output module configuration



process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuAnalysisSchedules_cff") 


## MC truth
process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff")
process.zMuMuSubskimOutputModule.outputCommands.extend(process.mcEventContent.outputCommands)
####

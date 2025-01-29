'''
Code to run scouting muon DQM, both to plot the distributions and efficiency for tag and
probe and also to plot distributions and efficiency of muon L1 seeds.

Author: Javier Garcia de Castro, email:javigdc@bu.edu
'''

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
## In line command options
options = VarParsing.VarParsing('analysis')
options.register('inputDataset',
                 '',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Input dataset")
options.register('runNumber',
                 379765,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Run to process")
options.register('minFile',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Minimum file to process")
options.register('maxFile',
                 10,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Maximum file to process")
options.register('output',
                 './',
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Output path")
options.parseArguments()

process = cms.Process("Demo")

#Load files
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_dataRun3_Prompt_v4')
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.MessageLogger.debugModules = cms.untracked.vstring('DQMGenericClient')
process.MessageLogger.cerr = cms.untracked.PSet(
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(10000)
    ),
 )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#Read files with the events
if options.inputDataset!='':
    listOfFiles = (os.popen("""dasgoclient -query="file dataset=%s instance=prod/global run=%i" """%(options.inputDataset, options.runNumber)).read()).split('\n')
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( listOfFiles[:-1] ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange('%i:1-%i:max'%(options.runNumber, options.runNumber))
)

from PhysicsTools.SelectorUtils.tools.vid_id_tools import *
#Load files
process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
process.gtStage2Digis.InputLabel = cms.InputTag( "hltFEDSelectorL1" )
process.load("HLTriggerOffline.Scouting.ScoutingMuonTagProbeAnalyzer_cfi")
process.load("HLTriggerOffline.Scouting.ScoutingMuonTriggerAnalyzer_cfi")
process.load("HLTriggerOffline.Scouting.ScoutingMuonMonitoring_Client_cff")
process.DQMStore = cms.Service("DQMStore")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")
process.dqmSaver.tag = '%i'%(options.minFile)

process.options = cms.untracked.PSet(numberOfThreads = cms.untracked.uint32(1))

#Sequence to be ran
process.allPath = cms.Path(process.scoutingMonitoringTagProbeMuonNoVtx
                           * process.scoutingMonitoringTagProbeMuonVtx
                           * process.muonEfficiencyNoVtx
                           * process.muonEfficiencyVtx 
                           * process.scoutingMonitoringTriggerMuon
                           * process.muonTriggerEfficiency)
#Save the files and close root file
process.p = cms.EndPath(process.dqmSaver)

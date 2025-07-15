import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

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

process.MessageLogger.cerr.FwkReport.reportEvery = 10000
# Enable LogInfo
process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    WARNING = cms.untracked.PSet(
        limit = cms.untracked.int32(0)
    ),
 )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
#        'root://cms-xrd-global.cern.ch///store/data/Run2024F/ScoutingPFRun3/HLTSCOUT/v1/000/383/779/00000/4059ea35-4366-4e47-a6ec-51b45f09b01f.root', # Scout dataset
        'root://cms-xrd-global.cern.ch///store/data/Run2024C/ScoutingPFMonitor/MINIAOD/PromptReco-v1/000/379/420/00000/780a6ec0-5061-4ffd-b86a-1a73aef0588a.root', # big one
    )
)


# process.load("EventFilter.L1TRawToDigi.gtStage2Digis_cfi")
# process.gtStage2Digis.InputLabel = cms.InputTag( "hltFEDSelectorL1" )

process.DQMStore = cms.Service("DQMStore")

process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")
process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
                                     fileName = cms.untracked.string("OUT_step1.root"))

#process.load("DQMServices.FileIO.DQMFileSaverPB_cfi")
process.dqmSaver.tag = 'SCOUTING'
process.load("HLTriggerOffline.Scouting.HLTScoutingEGammaDqmOffline_cff")

process.p = cms.Path(process.hltScoutingEGammaDqmOffline + process.dqmSaver)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
#process.p1 = cms.Path(cms.Sequence(process.scoutingEfficiencyHarvest + process.dqmSaver + process.dqmStoreStats))
process.schedule = cms.Schedule(process.p, process.DQMoutput_step)


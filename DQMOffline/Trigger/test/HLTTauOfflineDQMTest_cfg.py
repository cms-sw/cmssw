import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(2000)
        )


process.source = cms.Source("PoolSource",
               fileNames = cms.untracked.vstring(
        "/store/data/Run2012D/SingleMu/RAW-RECO/ZMu-PromptSkim-v1/000/207/454/00000/0246D2DC-4633-E211-B222-00261834B559.root",
        "/store/data/Run2012D/SingleMu/RAW-RECO/ZMu-PromptSkim-v1/000/207/454/00000/02ECD026-1C33-E211-8F36-20CF305B0534.root",
#        "/store/data/Run2012D/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/000/207/454/00000/08AB2AA5-9C33-E211-8EFE-00261894392F.root",
#        "/store/data/Run2012D/DoubleElectron/RAW-RECO/ZElectron-PromptSkim-v1/000/207/454/00000/08B8D682-9C33-E211-BCBC-003048FFCBB0.root",
                         )
                            )


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MessageLogger.HLTTauDQMOffline=dict()
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

# # remember to compile with USER_CXXFLAGS="-DEDM_ML_DEBUG"
# process.MessageLogger.cerr.threshold = cms.untracked.string("DEBUG")
# process.MessageLogger.debugModules.extend([
#         "hltTauOfflineMonitor_PFTaus",
#         "hltTauOfflineMonitor_Inclusive",
#         "HLTTauPostAnalysis_PFTaus",
#         "HLTTauPostAnalysis_Inclusive",
#         ])

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '') # for data
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '') # for MC

#process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Load Tau offline DQM
process.load("DQMOffline.Trigger.HLTTauDQMOffline_cff")

process.dqmEnv.subSystemFolder = "HLTOffline/HLTTAU"

#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/B/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)


process.p = cms.Path(process.HLTTauDQMOffline*process.dqmEnv)

process.o = cms.EndPath(process.HLTTauDQMOfflineHarvesting*process.HLTTauDQMOfflineQuality*process.dqmSaver)

process.schedule = cms.Schedule(process.p,process.o)

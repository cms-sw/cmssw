import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMOffline.EGamma.zmumugammaAnalyzer_cfi")
process.load("DQMOffline.EGamma.photonOfflineClient_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")

DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

 '/store/relval/CMSSW_6_0_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START53_V4-v1/0132/025C64FA-8FA2-E111-A72F-002618943869.root',
 '/store/relval/CMSSW_6_0_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START53_V4-v1/0132/5297CBFF-78A3-E111-A559-0018F3D09630.root',
 '/store/relval/CMSSW_6_0_0_pre5/RelValH130GGgluonfusion/GEN-SIM-RECO/START53_V4-v1/0132/8C258978-94A2-E111-AF3F-00248C0BE016.root'

))

from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)
#photonAnalysis.standAlone = cms.bool(True)

from DQMOffline.EGamma.photonOfflineClient_cfi import *
photonOfflineClient.standAlone = cms.bool(True)


#from DQMServices.Components.DQMStoreStats_cfi import *
#dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
#dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


#process.p1 = cms.Path(process.photonAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)
process.p1 = cms.Path(process.photonAnalysis*process.zmumugammaAnalysis*process.photonOfflineClient*process.dqmStoreStats)
#process.p1 = cms.Path(process.photonAnalysis*process.photonOfflineClient*process.dqmStoreStats)


process.schedule = cms.Schedule(process.p1)


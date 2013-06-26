import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.GeometryDB_cff")
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
	
'/store/relval/CMSSW_6_0_0_pre7-GR_R_53_V2_RelVal_zMu2011B/DoubleMu/RECO/v1/0000/124D426F-38BA-E111-AEC8-003048FFD75C.root'


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


import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMOffline.EGamma.zmumugammaAnalyzer_cfi")
process.load("DQMOffline.EGamma.photonOfflineClient_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")
process.load("FWCore.MessageLogger.MessageLogger_cfi")


DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

#process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
	
	'/store/relval/CMSSW_7_0_0_pre2/RelValZEE/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/0229B33C-E10F-E311-9C16-002618943829.root'
#	'/store/relval/CMSSW_7_0_0_pre2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/2EB245F1-A30F-E311-80ED-0025905938A4.root'

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

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger = cms.Service("MessageLogger")

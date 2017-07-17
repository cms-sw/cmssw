import FWCore.ParameterSet.Config as cms
process = cms.Process("photonDataCertification")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMOffline.EGamma.photonDataCertification_cfi")
process.load("DQMOffline.EGamma.photonOfflineClient_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")

DQMStore = cms.Service("DQMStore")

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(-1)
)

#process.MessageLogger.cerr.FwkReport.reportEvery = 100

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

#	'/store/relval/CMSSW_7_0_0_pre2/RelValZEE/GEN-SIM-DIGI-RECO/PRE_ST62_V8_FastSim-v1/00000/0229B33C-E10F-E311-9C16-002618943829.root'
	'/store/relval/CMSSW_7_0_0_pre2/RelValZEE/GEN-SIM-DIGI-RAW-HLTDEBUG/PRE_ST62_V8-v1/00000/3448C7FD-C90F-E311-A225-003048FFD76E.root'
							
))

from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)

from DQMOffline.EGamma.photonOfflineClient_cfi import *
photonOfflineClient.standAlone = cms.bool(True)

process.p = cms.Path(process.photonAnalysis*process.photonDataCertification*process.photonOfflineClient)
#process.p = cms.Path(process.dqmInfoEgamma*process.photonAnalysis*process.photonDataCertification)
#process.p = cms.Path(process.photonAnalysis)
process.schedule = cms.Schedule(process.p)

#print process.dumpPython()
#process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

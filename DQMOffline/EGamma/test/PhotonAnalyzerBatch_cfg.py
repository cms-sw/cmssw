import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")


process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMOffline.EGamma.zmumugammaAnalyzer_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
  #  input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'/store/relval/CMSSW_5_2_0_pre2/RelValZMM/GEN-SIM-RECO/START50_V9-v1/0214/C0CCF508-5244-E111-B211-00261894389E.root',
'/store/relval/CMSSW_5_2_0_pre2/RelValZMM/GEN-SIM-RECO/START50_V9-v1/0217/10161C44-D444-E111-A1AC-003048679164.root',
'/store/relval/CMSSW_5_2_0_pre2/RelValZMM/GEN-SIM-RECO/START50_V9-v1/0217/EC841A5E-A844-E111-84C1-0026189438DF.root'

))



from DQMOffline.EGamma.photonAnalyzer_cfi import *

photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.OutputFileName = cms.string('DQMOfflinePhotonsBatch.root')


from DQMServices.Components.DQMStoreStats_cfi import *

dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


process.p1 = cms.Path(process.photonAnalysis*process.zmumugammaAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)

process.schedule = cms.Schedule(process.p1)



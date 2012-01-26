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

    '/store/relval/CMSSW_4_4_2/RelValZMM/GEN-SIM-RECO/START44_V9_special_111130-v1/0067/2E329508-DE1D-E111-8EAC-0018F3D09658.root',
    '/store/relval/CMSSW_4_4_2/RelValZMM/GEN-SIM-RECO/START44_V9_special_111130-v1/0067/AA95995F-9B1B-E111-AB91-001A92971B16.root',
    '/store/relval/CMSSW_4_4_2/RelValZMM/GEN-SIM-RECO/START44_V9_special_111130-v1/0067/F471FC1D-951B-E111-AAA1-003048FFD740.root'
    
 #   '/store/relval/CMSSW_4_4_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START44_V9_special_111130-v1/0067/0C6A506C-D71C-E111-AF7F-003048FFCB84.root',
 #   '/store/relval/CMSSW_4_4_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START44_V9_special_111130-v1/0067/BCDB57FC-7C1D-E111-8CED-003048FFD754.root',
 #   '/store/relval/CMSSW_4_4_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START44_V9_special_111130-v1/0067/F26FCBAA-9C1B-E111-B890-001A92810AAA.root'

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



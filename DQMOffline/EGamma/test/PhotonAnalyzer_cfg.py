import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")

process.load("DQMOffline.EGamma.photonOfflineClient_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")


process.load("DQMServices.Components.DQMStoreStats_cfi")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(


        '/store/mc/Summer09/PhotonJet_Pt0to15/GEN-SIM-RECO/MC_31X_V2_preproduction_311-v1/0002/7C21B4CC-946C-DE11-B5FE-003048D4766A.root',
        '/store/mc/Summer09/PhotonJet_Pt0to15/GEN-SIM-RECO/MC_31X_V2_preproduction_311-v1/0000/C42E60D2-426C-DE11-AAB9-0015172561DA.root',
        '/store/mc/Summer09/PhotonJet_Pt0to15/GEN-SIM-RECO/MC_31X_V2_preproduction_311-v1/0000/9282DE91-426C-DE11-9DE3-003048D410BD.root',
        '/store/mc/Summer09/PhotonJet_Pt0to15/GEN-SIM-RECO/MC_31X_V2_preproduction_311-v1/0000/6C0BB2C4-426C-DE11-84B4-0015172C0910.root',
        '/store/mc/Summer09/PhotonJet_Pt0to15/GEN-SIM-RECO/MC_31X_V2_preproduction_311-v1/0000/600FA77D-426C-DE11-ABC3-0015172C0925.root',
        '/store/mc/Summer09/PhotonJet_Pt0to15/GEN-SIM-RECO/MC_31X_V2_preproduction_311-v1/0000/56D0D66E-426C-DE11-9138-0015172C0934.root'



))





from DQMOffline.EGamma.photonAnalyzer_cfi import *

photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)

from DQMOffline.EGamma.photonOfflineClient_cfi import *

photonOfflineClient.standAlone = cms.bool(True)

#from DQMServices.Components.DQMStoreStats_cfi import *

#dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
#dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


process.p1 = cms.Path(process.photonAnalysis*process.photonOfflineClient)
#process.p1 = cms.Path(process.photonAnalysis*process.photonOfflineClient*process.dqmStoreStats)

#process.p1 = cms.Path(process.photonAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)

process.schedule = cms.Schedule(process.p1)


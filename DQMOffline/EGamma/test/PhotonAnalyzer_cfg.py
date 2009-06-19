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

##             '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0008/E60958D5-0458-DE11-B0B7-000423D98E30.root',
##         '/store/relval/CMSSW_3_1_0_pre10/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0008/C0188C68-4157-DE11-AE7E-001D09F250AF.root'


        '/store/relval/CMSSW_3_1_0_pre9/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0006/6AA9BBFD-9F4E-DE11-8064-001617C3B79A.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0006/3A31C266-A14E-DE11-80AB-001D09F28F11.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0006/2E9E0AED-A34E-DE11-86D2-001D09F251D1.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0006/1E82DB64-B44E-DE11-880E-001D09F27003.root'



##             '/store/relval/CMSSW_3_1_0_pre9/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0007/82C99CCD-514F-DE11-998E-001D09F290CE.root',
##         '/store/relval/CMSSW_3_1_0_pre9/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0007/563F9E7B-EF4E-DE11-9603-000423D8F63C.root'




))





from DQMOffline.EGamma.photonAnalyzer_cfi import *

photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(True)

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


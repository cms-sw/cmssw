import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")


process.load("DQMOffline.EGamma.photonAnalyzer_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

            '/store/relval/CMSSW_3_1_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0004/DE2A9491-CD41-DE11-978D-001D09F2525D.root',
        '/store/relval/CMSSW_3_1_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0004/4074A697-E641-DE11-90A0-001D09F251D1.root',
        '/store/relval/CMSSW_3_1_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0004/1C1E7541-F241-DE11-BD00-001D09F23A20.root'


##         '/store/relval/CMSSW_3_1_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0004/EC62EB8A-E641-DE11-A4C6-001D09F2546F.root',
##         '/store/relval/CMSSW_3_1_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0004/6C21FC6B-F241-DE11-80B2-001D09F2AF1E.root',
##         '/store/relval/CMSSW_3_1_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0003/F2CADC90-7741-DE11-9E0B-001D09F29146.root',
##         '/store/relval/CMSSW_3_1_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0003/E84BCB6D-7141-DE11-B3EF-000423D94700.root',
##         '/store/relval/CMSSW_3_1_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0003/7EBDDFC9-7141-DE11-B583-001D09F23A6B.root',
##         '/store/relval/CMSSW_3_1_0_pre7/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0003/3002EFF9-7241-DE11-99B6-001D09F2983F.root'



))



from DQMOffline.EGamma.photonAnalyzer_cfi import *

photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.OutputFileName = cms.string('DQMOfflinePhotonsBatch.root')


from DQMServices.Components.DQMStoreStats_cfi import *

dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


process.p1 = cms.Path(process.photonAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)

process.schedule = cms.Schedule(process.p1)



import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.egammaDQMOffline_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


from DQMOffline.EGamma.photonAnalyzer_cfi import *
from DQMOffline.EGamma.piZeroAnalyzer_cfi import *

photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = cms.string('DQMOfflineRelValGammaJets_Pt_80_120.root')
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.useTriggerFiltering = cms.bool(True)

piZeroAnalysis.standAlone = cms.bool(True)
piZeroAnalysis.OutputMEsInRootFile = cms.bool(True)
piZeroAnalysis.OutputFileName = cms.string('DQMOfflineRelValGammaJets_Pt_80_120.root')
piZeroAnalysis.useTriggerFiltering = cms.bool(False)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

          '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/0CAA1EC0-F7F7-DD11-911D-000423D9970C.root',
         '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/58AF1BC2-06F8-DD11-91E3-000423D985B0.root',
         '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/DE195AF6-F8F7-DD11-A490-001617E30F56.root',
          '/store/relval/CMSSW_3_1_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0001/E4EDF9CD-F7F7-DD11-825B-00304879FA4A.root'

))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)


#process.p1 = cms.Path(process.egammaDQMOffline*process.MEtoEDMConverter*process.FEVT)
process.p1 = cms.Path(process.egammaDQMOffline)
process.schedule = cms.Schedule(process.p1)


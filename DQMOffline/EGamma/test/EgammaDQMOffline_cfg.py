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
        '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0002/C8A1B6A0-1733-DE11-B770-000423D98834.root'
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/80FD8D8D-1733-DE11-85C3-000423D6B444.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/6AB1F928-DC32-DE11-9BB6-001617C3B5D8.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/5AD5356A-DC32-DE11-B4D1-000423D9A212.root',
#        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/0231F05B-DC32-DE11-813B-001617C3B6CC.root'

))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)


#process.p1 = cms.Path(process.egammaDQMOffline*process.MEtoEDMConverter*process.FEVT)
process.p1 = cms.Path(process.egammaDQMOffline)
process.schedule = cms.Schedule(process.p1)


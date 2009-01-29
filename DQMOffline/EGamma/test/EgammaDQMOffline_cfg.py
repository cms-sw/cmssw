import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.egammaDQMOffline_cff")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)


from DQMOffline.EGamma.photonAnalyzer_cfi import *
from DQMOffline.EGamma.piZeroAnalyzer_cfi import *

photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = cms.string('DQMOfflineRelValGammaJets_Pt_80_120.root')
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.useTriggerFiltering = cms.bool(False)

piZeroAnalysis.standAlone = cms.bool(True)
piZeroAnalysis.OutputMEsInRootFile = cms.bool(True)
piZeroAnalysis.OutputFileName = cms.string('DQMOfflineRelValGammaJets_Pt_80_120.root')
piZeroAnalysis.useTriggerFiltering = cms.bool(False)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(



    '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/60FEA65F-2DDE-DD11-A12E-001617E30F56.root',
    '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/C0E99B59-34DE-DD11-9194-000423D174FE.root',
    '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/F41CD6A5-41DE-DD11-8049-000423D99AAA.root',
    '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/FCFD0B48-2BDE-DD11-97C8-000423D99B3E.root'
    

))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)


#process.p1 = cms.Path(process.photonAnalysis*process.MEtoEDMConverter)
process.p1 = cms.Path(process.egammaDQMOffline)
process.schedule = cms.Schedule(process.p1)


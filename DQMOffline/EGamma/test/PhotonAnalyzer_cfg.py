import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_2_2_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_V7_v1/0000/70BBDA00-1AAF-DD11-A7C3-001617DBD556.root',
        '/store/relval/CMSSW_2_2_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_V7_v1/0000/D0EDC0E8-CDAE-DD11-B7DD-001617C3B69C.root',
        '/store/relval/CMSSW_2_2_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_V7_v1/0000/E22B5AD8-CCAE-DD11-9D23-000423D98634.root',
        '/store/relval/CMSSW_2_2_0_pre1/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_V7_v1/0000/FE048742-CDAE-DD11-8A18-000423D9970C.root'



))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)

from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = 'DQMPhotonsStandaloneForMC.root'
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)
photonAnalysis.standAlone = cms.bool(True)



#process.p1 = cms.Path(process.MEtoEDMConverter)
#process.p1 = cms.Path(process.photonAnalysis*process.MEtoEDMConverter)
process.p1 = cms.Path(process.photonAnalysis)
process.schedule = cms.Schedule(process.p1)


import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
##        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/80FD8D8D-1733-DE11-85C3-000423D6B444.root',
##        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/6AB1F928-DC32-DE11-9BB6-001617C3B5D8.root',
##        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/5AD5356A-DC32-DE11-B4D1-000423D9A212.root',
##        '/store/relval/CMSSW_3_1_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0002/0231F05B-DC32-DE11-813B-001617C3B6CC.root'

        '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0002/F4802588-F232-DE11-A617-000423D94524.root',
        '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0002/AA4D1B30-DC32-DE11-92FC-000423D6006E.root',
        '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0002/60B7616F-1833-DE11-B131-000423D99614.root',
        '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0002/4A90C4C9-E832-DE11-AACB-000423D6CA02.root',
        '/store/relval/CMSSW_3_1_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_31X_v1/0002/2EEBC76D-DA32-DE11-AA9E-001617E30CA4.root'


#        '/store/relval/CMSSW_3_1_0_pre6/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_31X_v1/0002/C8A1B6A0-1733-DE11-B770-000423D98834.root'

))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)

from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = 'DQMPhotonsStandaloneForMC.root'
photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(True)
photonAnalysis.standAlone = cms.bool(True)



#process.p1 = cms.Path(process.MEtoEDMConverter)
#process.p1 = cms.Path(process.photonAnalysis*process.MEtoEDMConverter*process.FEVT)
process.p1 = cms.Path(process.photonAnalysis)
process.schedule = cms.Schedule(process.p1)

#process.outpath = cms.EndPath(process.FEVT)

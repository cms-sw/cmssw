import FWCore.ParameterSet.Config as cms
process = cms.Process("piZeroAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.piZeroAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(


        '/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0005/227BFA61-D5DD-DD11-88CE-001617C3B66C.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0005/780BFC49-41DE-DD11-ABB7-000423D99896.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0005/B6C941C9-D5DD-DD11-8F1A-001D09F25479.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0005/C670EC9F-D4DD-DD11-9476-000423D98930.root',
        '/store/relval/CMSSW_3_0_0_pre6/RelValQCD_Pt_80_120/GEN-SIM-RECO/IDEAL_30X_v1/0005/CC9494FF-D4DD-DD11-95A9-001D09F28EC1.root'


##    '/store/relval/CMSSW_3_0_0_pre6/RelValSingleGammaPt35/GEN-SIM-RECO/IDEAL_30X_v1/0005/98C45436-41DE-DD11-9B91-000423D95220.root'


##         '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/60FEA65F-2DDE-DD11-A12E-001617E30F56.root',
##         '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/C0E99B59-34DE-DD11-9194-000423D174FE.root',
##         '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/F41CD6A5-41DE-DD11-8049-000423D99AAA.root',
##         '/store/relval/CMSSW_3_0_0_pre6/RelValGammaJets_Pt_80_120/GEN-SIM-RECO/STARTUP_30X_v1/0005/FCFD0B48-2BDE-DD11-97C8-000423D99B3E.root'



))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)

from DQMOffline.EGamma.piZeroAnalyzer_cfi import *
piZeroAnalysis.OutputMEsInRootFile = cms.bool(True)
piZeroAnalysis.OutputFileName = 'DQMPiZeros.root'
piZeroAnalysis.Verbosity = cms.untracked.int32(0)
piZeroAnalysis.useTriggerFiltering = cms.bool(False)
piZeroAnalysis.standAlone = cms.bool(True)



#process.p1 = cms.Path(process.MEtoEDMConverter)
#process.p1 = cms.Path(process.piZeroAnalysis*process.MEtoEDMConverter)
process.p1 = cms.Path(process.piZeroAnalysis)
process.schedule = cms.Schedule(process.p1)


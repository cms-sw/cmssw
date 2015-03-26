import FWCore.ParameterSet.Config as cms
process = cms.Process("piZeroAnalysis")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("DQMOffline.EGamma.piZeroAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_6_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V24-v1/0001/364E7B38-6F27-DF11-91A9-0026189438D4.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V24-v1/0000/48AE643B-0727-DF11-99FB-001731AF66F5.root'


       

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
piZeroAnalysis.standAlone = cms.bool(True)



#process.p1 = cms.Path(process.MEtoEDMConverter)
#process.p1 = cms.Path(process.piZeroAnalysis*process.MEtoEDMConverter)
process.p1 = cms.Path(process.piZeroAnalysis)
process.schedule = cms.Schedule(process.p1)


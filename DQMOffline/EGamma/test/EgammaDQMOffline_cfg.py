import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
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
piZeroAnalysis.OutputMEsInRootFile = cms.bool(False)
piZeroAnalysis.OutputFileName = cms.string('DQMOfflineRelValGammaJets_Pt_80_120.root')


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_6_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V24-v1/0001/364E7B38-6F27-DF11-91A9-0026189438D4.root',
        '/store/relval/CMSSW_3_6_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V24-v1/0000/48AE643B-0727-DF11-99FB-001731AF66F5.root'

))



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)


#process.p1 = cms.Path(process.egammaDQMOffline*process.MEtoEDMConverter*process.FEVT)
process.p1 = cms.Path(process.egammaDQMOffline)
process.schedule = cms.Schedule(process.p1)


import FWCore.ParameterSet.Config as cms
process = cms.Process("TestPhotonValidator")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("DQMOffline.EGamma.cosmicPhotonAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
#    input = cms.untracked.int32(5000)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_2_6/RelValCosmics/RECO/CRAFT09_R_V2-v2/0001/F8CD7049-C29B-DE11-86CB-000423D98920.root'
    )
)

#process.FEVT = cms.OutputModule("PoolOutputModule",
#    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
#    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
#)


from DQMOffline.EGamma.photonAnalyzer_cfi import *
photonAnalysis.OutputMEsInRootFile = cms.bool(True)
photonAnalysis.OutputFileName = cms.string('DQMOfflineCRAFT.root')
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.useTriggerFiltering = cms.bool(False)



process.p1 = cms.Path(process.cosmicPhotonAnalysis)
process.schedule = cms.Schedule(process.p1)

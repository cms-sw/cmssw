import FWCore.ParameterSet.Config as cms
process = cms.Process("TestPhotonValidator")

process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("DQMOffline.EGamma.cosmicPhotonAnalyzer_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(5000)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       "file:reco2.root"
    )
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('photonsMEtoEDMConverter.root')
)

process.p1 = cms.Path(process.cosmicPhotonAnalysis*process.MEtoEDMConverter)
process.outPath = cms.EndPath(process.FEVT)

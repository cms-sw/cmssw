import FWCore.ParameterSet.Config as cms

process = cms.Process("TestElectronValidator")
process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")

process.load("DQMOffline.EGamma.electronAnalyzer_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("EgammaAnalysis.EgammaIsolationProducers.egammaSuperClusterMerger_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(20)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValSingleElectronPt35-1213987236-IDEAL_V2-2nd/0004/5233133B-C640-DD11-A56C-000423D6CA02.root',)
#    '/store/relval/2008/6/2/RelVal-RelValSingleElectronPt35-1212355159-IDEAL_V1-2nd/0000/02FE32F3-A630-DD11-868D-001617E30F4C.root', 
#        '/store/relval/2008/6/2/RelVal-RelValSingleElectronPt35-1212355159-IDEAL_V1-2nd/0000/9CD77E8E-A630-DD11-9C8B-000423D98DD4.root', 
#        '/store/relval/2008/6/2/RelVal-RelValSingleElectronPt35-1212355159-IDEAL_V1-2nd/0000/D071B9F8-A630-DD11-A840-000423D6CA6E.root')
)

process.DQMStore = cms.Service("DQMStore")

process.mergedSuperClusters = cms.EDFilter("SuperClusterMerger",
    src = cms.VInputTag(cms.InputTag("correctedHybridSuperClusters"), cms.InputTag("correctedMulti5x5SuperClustersWithPreshower"))
)

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *_MEtoEDMConverter_*_*'),
    fileName = cms.untracked.string('MEtoEDMConverter.root')
)

process.p1 = cms.Path(process.mergedSuperClusters*process.gsfElectronAnalysis*process.MEtoEDMConverter)
process.outpath = cms.EndPath(process.FEVT)
process.schedule = cms.Schedule(process.p1,process.outpath)




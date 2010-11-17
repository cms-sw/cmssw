import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")


process.load("DQMOffline.EGamma.photonAnalyzer_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")
process.load("DQMServices.Components.DQMStoreStats_cfi")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'file:/pscratch/ndcms/bestman/storage/cms/tauAnalysis/RelVelZEE_38X.root'


))



from DQMOffline.EGamma.photonAnalyzer_cfi import *

photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)
photonAnalysis.standAlone = cms.bool(True)
photonAnalysis.OutputFileName = cms.string('DQMOfflinePhotonsBatch.root')


from DQMServices.Components.DQMStoreStats_cfi import *

dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


process.p1 = cms.Path(process.photonAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)

process.schedule = cms.Schedule(process.p1)



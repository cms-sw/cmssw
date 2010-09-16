import FWCore.ParameterSet.Config as cms
process = cms.Process("photonAnalysis")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("DQMOffline.EGamma.photonAnalyzer_cfi")

process.load("DQMOffline.EGamma.photonOfflineClient_cfi")

process.load("DQMServices.Components.MEtoEDMConverter_cff")


process.load("DQMServices.Components.DQMStoreStats_cfi")


DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(

'file:/afs/crc.nd.edu/group/NDCMS/data01/RelVal370pre3Photons1.root',
'file:/afs/crc.nd.edu/group/NDCMS/data01/RelVal370pre3Photons2.root',
'file:/afs/crc.nd.edu/group/NDCMS/data01/RelVal370pre3Photons3.root'

))





from DQMOffline.EGamma.photonAnalyzer_cfi import *

photonAnalysis.Verbosity = cms.untracked.int32(0)
photonAnalysis.useTriggerFiltering = cms.bool(False)

from DQMOffline.EGamma.photonOfflineClient_cfi import *

photonOfflineClient.standAlone = cms.bool(True)

#from DQMServices.Components.DQMStoreStats_cfi import *

#dqmStoreStats.runOnEndRun = cms.untracked.bool(False)
#dqmStoreStats.runOnEndJob = cms.untracked.bool(True)


process.p1 = cms.Path(process.photonAnalysis*process.photonOfflineClient)
#process.p1 = cms.Path(process.photonAnalysis*process.photonOfflineClient*process.dqmStoreStats)

#process.p1 = cms.Path(process.photonAnalysis)
#process.p1 = cms.Path(process.photonAnalysis*process.dqmStoreStats)

process.schedule = cms.Schedule(process.p1)


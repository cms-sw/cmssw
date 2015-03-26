import FWCore.ParameterSet.Config as cms
process = cms.Process("photonOfflineClient")

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

process.load("DQMOffline.EGamma.photonOfflineClient_cfi")


process.DQMStore = cms.Service("DQMStore")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

from DQMOffline.EGamma.photonOfflineClient_cfi import *

photonOfflineClient.batch = cms.bool(True)


process.source = cms.Source("EmptySource"
)


process.p1 = cms.Path(process.photonOfflineClient)
process.schedule = cms.Schedule(process.p1)



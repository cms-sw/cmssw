import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonOfflineClient_cfi import *
import DQMOffline.EGamma.photonOfflineClient_cfi


stdPhotonOfflineClient = DQMOffline.EGamma.photonOfflineClient_cfi.photonOfflineClient.clone()
stdPhotonOfflineClient.ComponentName = cms.string('stdPhotonOfflineClient')
stdPhotonOfflineClient.analyzerName = cms.string('stdPhotonAnalyzer')

photonOfflineDQMClient = cms.Sequence(photonOfflineClient*stdPhotonOfflineClient)

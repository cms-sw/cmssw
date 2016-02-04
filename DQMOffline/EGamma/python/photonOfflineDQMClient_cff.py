import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonOfflineClient_cfi import *

photonOfflineDQMClient = cms.Sequence(photonOfflineClient)

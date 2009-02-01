import FWCore.ParameterSet.Config as cms

from DQMOffline.Ecal.ecalOfflineCosmicClient_cfi import *

ecalOfflineCosmicClientSequence = cms.Sequence(ecalOfflineCosmicClient)

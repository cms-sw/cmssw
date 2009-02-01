import FWCore.ParameterSet.Config as cms

from DQMOffline.Ecal.ecalOfflineCosmicTask_cfi import *

ecalOfflineCosmicTaskSequence = cms.Sequence(ecalOfflineCosmicTask)

import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonOfflineClient_cfi import *
from DQMOffline.EGamma.electronOfflineClientSequence_cff import *


egammaPostProcessing = cms.Sequence(photonOfflineClient*electronOfflineClientSequence)

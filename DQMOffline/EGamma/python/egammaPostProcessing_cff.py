import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.photonOfflineClient_cfi import *
from DQMOffline.EGamma.electronClientSequence_cff import *


egammaPostprocessing = cms.Sequence(photonOfflineClient,electronClientSequence)

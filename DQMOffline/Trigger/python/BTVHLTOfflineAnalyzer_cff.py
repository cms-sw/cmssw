import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BTVHLTOfflineSource_cfi import *

BTVHLTOfflineAnalyzer = cms.Sequence(
   BTVHLTOfflineSource
)

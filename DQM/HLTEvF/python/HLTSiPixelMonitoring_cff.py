import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_cff import *

pixelOnlineMonitorHLTsequence = cms.Sequence(
    sipixelMonitorHLTsequence
)

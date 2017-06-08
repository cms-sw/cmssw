import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.SiPixel_OfflineMonitoring_Client_cff import *

pixelOnlineHarvesterHLTsequence = cms.Sequence(
    sipixelHarvesterHLTsequence
)

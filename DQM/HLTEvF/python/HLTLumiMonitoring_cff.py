import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DQMOffline_LumiMonitoring_cff import *

lumiOnlineMonitorHLTsequence = cms.Sequence(
    lumiMonitorHLTsequence
)

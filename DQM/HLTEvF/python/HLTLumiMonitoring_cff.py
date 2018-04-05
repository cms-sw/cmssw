import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DQMOffline_LumiMontiroring_cff import *

lumiOnlineMonitorHLTsequence = cms.Sequence(
    lumiMonitorHLTsequence
)

import FWCore.ParameterSet.Config as cms

from DQM.HLTEvF.HLTTrackingMonitoring_Client_cff import *
from DQM.HLTEvF.HLTSiPixelMonitoring_Client_cff import *

client = cms.EndPath(
    trackingMonitorClientHLT
    + trackingForElectronsMonitorClientHLT
    + pixelOnlineHarvesterHLTsequence
)

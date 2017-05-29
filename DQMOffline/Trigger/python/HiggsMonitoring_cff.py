import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.PhotonMonitor_cff import *

higgsMonitorHLT = cms.Sequence(
    higgsHLTDiphotonMonitoring
)

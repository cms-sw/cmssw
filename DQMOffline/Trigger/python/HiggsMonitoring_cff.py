import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitor_cff import *


higgsMonitorHLT = cms.Sequence(
   mssmHbbMonitor
)

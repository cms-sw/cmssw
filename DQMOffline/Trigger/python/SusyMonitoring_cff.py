import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.RazorMonitor_cff import *
from DQMOffline.Trigger.VBFSUSYMonitor_cff import *
from DQMOffline.Trigger.LepHTMonitor_cff import *

susyMonitorHLT = cms.Sequence(
    susyHLTRazorMonitoring
  + susyHLTVBFMonitoring
  + LepHTMonitor 
)

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.HCALMonitoring_cff import *

hcalOnlineMonitoringSequence = cms.Sequence(
    hcalMonitoringSequence
)

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbMonitor_Client_cfi import *

higgsClient = cms.Sequence(
   mssmHbbBtagTriggerEfficiency
)

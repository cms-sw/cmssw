import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.MssmHbbBtagTriggerMonitor_Client_cfi import *

higgsClient = cms.Sequence(
   mssmHbbBtagTriggerEfficiency
)

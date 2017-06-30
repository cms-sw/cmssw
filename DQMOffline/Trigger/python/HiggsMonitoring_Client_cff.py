import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDHarvester import DQMEDHarvester
from DQMOffline.Trigger.MssmHbbMonitor_Client_cfi import mssmHbbBtagTriggerEfficiency

mssmHbbClient = cms.Sequence(
   mssmHbbBtagTriggerEfficiency
)

import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.RazorMonitor_Client_cff import *

susyClient = cms.Sequence(
               susyHLTRazorClient
)

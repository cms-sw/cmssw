import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.VBFSUSYMonitor_Client_cff import *
from DQMOffline.Trigger.LepHTMonitor_cff import *
from DQMOffline.Trigger.susyHLTEleCaloJetsClient_cfi import *

susyClient = cms.Sequence(
    vbfsusyClient
  + LepHTClient
  + susyHLTEleCaloJetsClient
)

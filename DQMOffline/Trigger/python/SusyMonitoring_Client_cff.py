import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.VBFSUSYMonitor_Client_cff import *

susyClient = cms.Sequence(
    vbfsusyClient
)

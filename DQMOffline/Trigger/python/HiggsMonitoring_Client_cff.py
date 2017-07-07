import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.VBFMETMonitor_Client_cff import *

higgsClient = cms.Sequence(
    vbfmetClient
)

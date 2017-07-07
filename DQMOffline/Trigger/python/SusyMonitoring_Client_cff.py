import FWCore.ParameterSet.Config as cms
from DQMOffline.Trigger.LepHTMonitor_cff import *

susyClient = cms.Sequence(
        LepHTClient
)

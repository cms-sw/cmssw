import FWCore.ParameterSet.Config as cms
from DQMOffline.Trigger.HTMonitoring_Client_cff import *
from DQMOffline.Trigger.METMonitoring_Client_cff import *

exoticaClient = cms.Sequence(
    htClient *
    metClient
)

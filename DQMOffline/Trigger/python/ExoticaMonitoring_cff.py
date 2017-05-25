import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cff import *
from DQMOffline.Trigger.PhotonMonitor_cff import *

exoticaMonitorHLT = cms.Sequence(
    exoHLTMETmonitoring
    + exoHLTPhotonmonitoring
)

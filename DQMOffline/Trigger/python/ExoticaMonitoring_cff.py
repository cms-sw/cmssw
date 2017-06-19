import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.METMonitor_cff import *
from DQMOffline.Trigger.DisplacedJetHTMonitor_cff import *
from DQMOffline.Trigger.DisplacedJetTrackMonitor_cff import *

exoticaMonitorHLT = cms.Sequence(
    exoHLTMETmonitoring
   +exoHLTDJHTmonitoring
   +exoHLTDJTrackmonitoring
)

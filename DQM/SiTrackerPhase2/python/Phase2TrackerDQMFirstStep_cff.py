import FWCore.ParameterSet.Config as cms
from DQM.SiTrackerPhase2.Phase2TrackerMonitorDigi_cff import *
from DQM.SiTrackerPhase2.Phase2ITMonitorRecHit_cff import *

trackerphase2DQMSource = cms.Sequence( pixDigiMon 
                                       + otDigiMon
                                       +rechitMonitorIT
)

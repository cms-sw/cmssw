import FWCore.ParameterSet.Config as cms

from DQM.Phase2OuterTracker.OuterTrackerMonitorTTCluster_cfi import *
from DQM.Phase2OuterTracker.OuterTrackerMonitorTTStub_cfi import *
#from DQM.Phase2OuterTracker.OuterTrackerMonitorTTTrack_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerMonitorTTCluster
                                 * OuterTrackerMonitorTTStub
#                                 * OuterTrackerMonitorTTTrack
                                 )

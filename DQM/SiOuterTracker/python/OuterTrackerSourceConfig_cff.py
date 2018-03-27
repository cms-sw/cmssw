import FWCore.ParameterSet.Config as cms

from DQM.SiOuterTracker.OuterTrackerMonitorTTCluster_cfi import *
from DQM.SiOuterTracker.OuterTrackerMonitorTTStub_cfi import *
from DQM.SiOuterTracker.OuterTrackerMonitorTTTrack_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerMonitorTTCluster
                                 * OuterTrackerMonitorTTStub
                                 * OuterTrackerMonitorTTTrack
                                 )

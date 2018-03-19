import FWCore.ParameterSet.Config as cms

from DQM.OuterTrackerL1.OuterTrackerMonitorTTCluster_cfi import *
from DQM.OuterTrackerL1.OuterTrackerMonitorTTStub_cfi import *
from DQM.OuterTrackerL1.OuterTrackerMonitorTTTrack_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerMonitorTTCluster
                                 * OuterTrackerMonitorTTStub
                                 * OuterTrackerMonitorTTTrack
                                 )

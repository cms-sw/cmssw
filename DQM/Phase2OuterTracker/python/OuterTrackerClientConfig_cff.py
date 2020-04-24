import FWCore.ParameterSet.Config as cms

from DQM.Phase2OuterTracker.OuterTrackerMonitorTTClusterClient_cfi import *
from DQM.Phase2OuterTracker.OuterTrackerMonitorTTStubClient_cfi import *
#from DQM.Phase2OuterTracker.OuterTrackerMonitorTTTrackClient_cfi import *

OuterTrackerClient = cms.Sequence(OuterTrackerMonitorTTClusterClient
                                 * OuterTrackerMonitorTTStubClient
#                                 * OuterTrackerMonitorTTTrackClient
                                 )


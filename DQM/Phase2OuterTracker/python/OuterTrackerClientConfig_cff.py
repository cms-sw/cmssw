import FWCore.ParameterSet.Config as cms

from DQM.Phase2OuterTracker.OuterTrackerMonitorClusterClient_cfi import *
from DQM.Phase2OuterTracker.OuterTrackerMonitorStubClient_cfi import *
from DQM.Phase2OuterTracker.OuterTrackerMonitorTrackClient_cfi import *

OuterTrackerClient = cms.Sequence(OuterTrackerMonitorClusterClient *
				  OuterTrackerMonitorStubClient *
				  OuterTrackerMonitorTrackClient )


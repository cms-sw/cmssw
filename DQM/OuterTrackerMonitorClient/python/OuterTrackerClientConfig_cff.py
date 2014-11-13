import FWCore.ParameterSet.Config as cms

from DQM.OuterTrackerMonitorClient.OuterTrackerMonitorClusterClient_cfi import *
from DQM.OuterTrackerMonitorClient.OuterTrackerMonitorStubClient_cfi import *

OuterTrackerClient = cms.Sequence(OuterTrackerMonitorClusterClient *
				  OuterTrackerMonitorStubClient)


import FWCore.ParameterSet.Config as cms

from DQM.OuterTrackerMonitorClient.OuterTrackerMonitorClusterClient_cfi import *

OuterTrackerClient = cms.Sequence(OuterTrackerMonitorClusterClient)


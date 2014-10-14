import FWCore.ParameterSet.Config as cms

from DQM.OuterTrackerMonitorCluster.OuterTrackerMonitorCluster_cfi import *
from DQM.OuterTrackerMonitorSimulation.OuterTrackerMonitorSimulation_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerMonitorCluster *
                                  OuterTrackerMonitorSimulation)

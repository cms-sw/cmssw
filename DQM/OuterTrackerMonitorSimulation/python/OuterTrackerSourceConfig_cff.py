import FWCore.ParameterSet.Config as cms

from DQM.OuterTrackerMonitorCluster.OuterTrackerMonitorSimulation_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerMonitorSimulation)

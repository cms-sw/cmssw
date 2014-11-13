import FWCore.ParameterSet.Config as cms

from Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi import *
from DQM.Phase2OuterTracker.OuterTrackerMonitorCluster_cfi import *
from DQM.Phase2OuterTracker.OuterTrackerMonitorStub_cfi import *

OuterTrackerSource = cms.Sequence(OuterTrackerMonitorCluster * 
				  OuterTrackerMonitorStub)

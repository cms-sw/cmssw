import FWCore.ParameterSet.Config as cms

# Reconstruction geometry services
#  Tracking Geometry
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *

# Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# TrackerAdditionalParametersPerDet contains only default values, needed for consistency with Phase 2
from Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi import *

# Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *

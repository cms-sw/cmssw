import FWCore.ParameterSet.Config as cms

# Reconstruction geometry services
#  Tracking Geometry
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *

# Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *

# Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *

import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generateRun4Geometry.py
# If you notice a mistake, please update the generating script, not just this config

from Configuration.Geometry.GeometryExtendedRun4D500_cff import *

# tracker
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cff import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = True


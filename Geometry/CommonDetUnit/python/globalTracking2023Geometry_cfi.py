import FWCore.ParameterSet.Config as cms

#
# This cfi is should be included to build the GlobalTrackingGeometry with all 
# concrete TrackingGeometries for muon and tracker.
#
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumbering2023Geometry_cfi import *
from Geometry.TrackerGeometryBuilder.trackerSLHCGeometry_cfi import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *


import FWCore.ParameterSet.Config as cms

#
# This cfi is should be included to build the GlobalTrackingGeometry with all 
# concrete TrackingGeometries for muon and tracker.
#
from Geometry.CSCGeometryBuilder.cscGeometryDB_cfi import *
from Geometry.RPCGeometryBuilder.rpcGeometryDB_cfi import *
from Geometry.DTGeometryBuilder.dtGeometryDB_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometryDB_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometryDB_cfi import *
from Geometry.CommonTopologies.bareGlobalTrackingGeometry_cfi import *

# foo bar baz
# 4oM0xOjJw9Ye1
# 3IzShbPdhpZpl

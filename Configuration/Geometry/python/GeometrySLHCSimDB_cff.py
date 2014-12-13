import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from GeometryReaders.XMLIdealGeometryESSource.cmsGeometryDB_cff import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumbering2023GeometryDB_cfi import *

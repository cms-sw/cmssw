import FWCore.ParameterSet.Config as cms
from FastSimulation.Configuration.Geometries_cff import _fastSimGeometryCustoms
from Geometry.TrackerGeometryBuilder.trackerGeometryDB_cfi import trackerGeometryDB as _trackerGeometry
from Geometry.DTGeometryBuilder.dtGeometryDB_cfi import DTGeometryESModule as _DTGeometryESModule
from Geometry.CSCGeometryBuilder.cscGeometryDB_cfi import CSCGeometryESModule as _CSCGeometryESModule

def _fastSimGeometryCustomsDB(process):
    _fastSimGeometryCustoms(process, _trackerGeometry, _DTGeometryESModule, _CSCGeometryESModule, 'DB')

from Configuration.Eras.Modifier_fastSim_cff import fastSim
modifyGeomDB_fastSim = fastSim.makeProcessModifier(_fastSimGeometryCustomsDB)

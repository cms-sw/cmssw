import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

def _loadSimGeomDBDDD(process) :
    process.load('Configuration.Geometry.GeometryDDDSimDB_cff')

def _loadSimGeomDBDD4hep(process) :
    process.load('Configuration.Geometry.GeometryDD4hepSimDB_cff')

modifyGeometryConfiguration = (~dd4hep).makeProcessModifier(_loadSimGeomDBDDD)
modifyGeometryConfiguration_dd4hep = dd4hep.makeProcessModifier(_loadSimGeomDBDD4hep)

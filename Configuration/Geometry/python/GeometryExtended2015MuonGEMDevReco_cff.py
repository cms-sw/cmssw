import FWCore.ParameterSet.Config as cms

# Ideal geometry, needed for transient ECAL alignement
from Configuration.Geometry.GeometryExtended2015MuonGEMDev_cff import *
from Configuration.Geometry.GeometryReco_cff import *

from Geometry.GEMGeometryBuilder.gemGeometry_cfi import *
from Geometry.GEMGeometryBuilder.me0Geometry_cfi import *
from Geometry.RPCGeometryBuilder.rpcGeometry_cfi import *

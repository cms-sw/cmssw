import FWCore.ParameterSet.Config as cms

#
# This cfi should be included to build the RPC geometry model.
#
from Geometry.RPCGeometry.rpcGeometry_cfi import *
RPCGeometryESModule.compatibiltyWith11 = False


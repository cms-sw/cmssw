import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedFlat_cfi import *
#
# sets limits to << 1 um
#
VtxSmeared.MinX = -0.00000001
VtxSmeared.MaxX = 0.00000001
VtxSmeared.MinY = -0.00000001
VtxSmeared.MaxY = 0.00000001
VtxSmeared.MinZ = -0.00000001
VtxSmeared.MaxZ = 0.00000001

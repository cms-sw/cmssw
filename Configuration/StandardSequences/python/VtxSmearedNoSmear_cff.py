import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedFlat_cfi import *
#
# sets limits to << 1 um
#
VertexSmearingParameters.MinX = -0.00000001
VertexSmearingParameters.MaxX = 0.00000001
VertexSmearingParameters.MinY = -0.00000001
VertexSmearingParameters.MaxY = 0.00000001
VertexSmearingParameters.MinZ = -0.00000001
VertexSmearingParameters.MaxZ = 0.00000001

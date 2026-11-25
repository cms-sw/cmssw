import FWCore.ParameterSet.Config as cms

# BTL test beam 2025 (DUT behind REF): DUT and REF BTL modules both active, standalone world (no ETL)

from Geometry.MTDTBCommonData.GeometryDD4hepExtendedBTLTB2025_DUTbehind_cff import *

# timing
from RecoMTD.DetLayers.mtdDetLayerGeometry_cfi import *
from Geometry.MTDGeometryBuilder.mtdParameters_cff import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *
mtdNumberingGeometry.isTB = True
from Geometry.MTDNumberingBuilder.mtdTopology_cfi import *
from Geometry.MTDGeometryBuilder.mtdGeometry_cfi import *
from Geometry.MTDGeometryBuilder.idealForDigiMTDGeometry_cff import *
mtdGeometry.applyAlignment = False
# only two BTL modules in the test beam geometry
mtdTopology.isFull = False

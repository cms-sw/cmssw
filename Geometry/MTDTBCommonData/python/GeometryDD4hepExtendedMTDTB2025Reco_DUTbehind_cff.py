import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generateRun4Geometry.py
# If you notice a mistake, please update the generating script, not just this config

from Geometry.MTDTBCommonData.GeometryDD4hepExtendedMTDTB2025_DUTbehind_cff import *

# timing
from RecoMTD.DetLayers.mtdDetLayerGeometry_cfi import *
from Geometry.MTDGeometryBuilder.mtdParameters_cff import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *
from Geometry.MTDNumberingBuilder.mtdTopology_cfi import *
from Geometry.MTDGeometryBuilder.mtdGeometry_cfi import *
from Geometry.MTDGeometryBuilder.idealForDigiMTDGeometry_cff import *
mtdGeometry.applyAlignment = False


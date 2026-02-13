import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generateRun4Geometry.py
# If you notice a mistake, please update the generating script, not just this config

from Configuration.Geometry.GeometryDD4hep_cff import *
DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("Geometry/MTDTBCommonData/data/dd4hep/cmsExtendedGeometryMTDTB2025_DUTinFront.xml")

from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *

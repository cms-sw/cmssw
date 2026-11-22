import FWCore.ParameterSet.Config as cms

# BTL test beam 2025 (DUT behind REF): DUT and REF BTL modules both active, standalone world (no ETL)

from Configuration.Geometry.GeometryDD4hep_cff import *
DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("Geometry/MTDTBCommonData/data/dd4hep/cmsExtendedGeometryBTLTB2025_DUTbehind.xml")

from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *
mtdNumberingGeometry.isTB = True

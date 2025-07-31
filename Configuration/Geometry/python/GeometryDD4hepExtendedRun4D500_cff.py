import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generateRun4Geometry.py
# If you notice a mistake, please update the generating script, not just this config

from Configuration.Geometry.GeometryDD4hep_cff import *
DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometryRun4D500.xml")

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff import *

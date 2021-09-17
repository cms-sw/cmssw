import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generate2021Geometry.py
# If you notice a mistake, please update the generating script, not just this config

from Configuration.Geometry.GeometryDD4hep_cff import *
DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021ZeroMaterial.xml")

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MuonNumbering.muonOffsetESProducer_cff import *


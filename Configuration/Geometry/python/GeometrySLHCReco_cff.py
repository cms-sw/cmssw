import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.Geometry.Phase1_cmsSimIdealGeometryXML_cfi import *

# Reconstruction geometry services
#  Tracking Geometry
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
trackerGeometry.applyAlignment = cms.bool(False)

#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *

#Muon
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.AlignedCaloGeometryDBReader_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
from Geometry.HcalCommonData.hcalDDDRecConstants_cfi import *
from Geometry.HcalEventSetup.hcalTopologyIdealSLHC_cfi import *

## Just in case if a wrong DB record is pulled in:
##
es_prefer_ZdcEP = cms.ESPrefer("ZdcHardcodeGeometryEP")
es_prefer_HcalEP = cms.ESPrefer("HcalHardcodeGeometryEP")
es_prefer_TrackerEP = cms.ESPrefer("TrackerGeometricDetESModule", "trackerNumberingGeometry")
es_prefer_CaloTowerEP = cms.ESPrefer("CaloTowerHardcodeGeometryEP")
es_prefer_EcalBarrelEP = cms.ESPrefer("EcalBarrelGeometryEP")
es_prefer_EcalEndcapEP = cms.ESPrefer("EcalEndcapGeometryEP")
es_prefer_EcalPreshowerEP = cms.ESPrefer("EcalPreshowerGeometryEP")
es_prefer_CastorEP = cms.ESPrefer("CastorHardcodeGeometryEP")

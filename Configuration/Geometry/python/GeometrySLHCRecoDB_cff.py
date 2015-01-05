import FWCore.ParameterSet.Config as cms

#  Tracking Geometry
from Geometry.CommonDetUnit.globalTracking2023GeometryDB_cfi import *

#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumbering2023GeometryDB_cfi import *
trackerSLHCGeometryDB.applyAlignment = cms.bool(False)
#
# When there will be an alignment, perhaps, it will use a label 
#trackerSLHCGeometryDB.alignmentsLabel = cms.string('fakeForIdeal')

#Muon
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.AlignedCaloGeometryDBReader_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerSLHCGeometryDB_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometryDB_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometryDB_cff import *


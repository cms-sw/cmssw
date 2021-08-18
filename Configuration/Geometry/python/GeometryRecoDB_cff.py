import FWCore.ParameterSet.Config as cms

#  Tracking Geometry
from Geometry.CommonTopologies.globalTrackingGeometryDB_cfi import *

#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
# TrackerAdditionalParametersPerDet contains only default values, needed for consistency with Phase 2
from Geometry.TrackerGeometryBuilder.TrackerAdditionalParametersPerDet_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *

#Muon
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.AlignedCaloGeometryDBReader_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
from Geometry.HcalCommonData.hcalDBConstants_cff import *
from Geometry.HcalEventSetup.hcalTopologyIdeal_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometryDB_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometryDB_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometryDB_cff import *

# GEM present from 2017 onwards

def _loadGeometryESProducers( theProcess ) :
   theProcess.load('Geometry.GEMGeometryBuilder.gemGeometryDB_cfi')

from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
modifyGeometryConfigurationRun2_cff_ = run2_GEM_2017.makeProcessModifier( _loadGeometryESProducers )

from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
modifyGeometryConfigurationRun3_cff_ = run3_GEM.makeProcessModifier( _loadGeometryESProducers )

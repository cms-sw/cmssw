import FWCore.ParameterSet.Config as cms

# Ideal geometry, needed for transient ECAL alignement
from Configuration.Geometry.GeometryExtended2023SHCalNoTaper4Eta_cff import *

# Reconstruction geometry services
#  Tracking Geometry
#bah - well, this is not a cfi!
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *

#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *

#Muon
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.GEMGeometryBuilder.gemGeometry_cfi import *
from Geometry.GEMGeometryBuilder.me0Geometry_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *
trackerGeometry.applyAlignment = cms.bool(False)

#  Calorimeters
from Geometry.HGCalCommonData.shashlikNumberingInitialization_cfi import *
from Geometry.CaloEventSetup.ShashlikTopology_cfi import *
from Geometry.FCalGeometry.ShashlikGeometryESProducer_cfi import *

from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometryBuilder_cfi import *

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL'          , 
                                'ZDC'           ,
                                'CASTOR'        ,
                                'EcalBarrel'    , 
                                'Shashlik',
                                'TOWER'           )
)

from Geometry.EcalAlgo.EcalBarrelGeometry_cfi import *
from Geometry.HcalEventSetup.HcalGeometry_cfi import *
from Geometry.HcalEventSetup.CaloTowerGeometry_cfi import *
from Geometry.HcalEventSetup.CaloTowerTopology_cfi import *
from Geometry.HcalCommonData.hcalDDDRecConstants_cfi import *
from Geometry.HcalEventSetup.hcalTopologyIdealSLHC_cfi import *
from Geometry.ForwardGeometry.ForwardGeometry_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *

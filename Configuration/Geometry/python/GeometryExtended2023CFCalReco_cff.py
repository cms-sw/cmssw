import FWCore.ParameterSet.Config as cms

# Ideal geometry, needed for transient ECAL alignement
from Configuration.Geometry.GeometryExtended2023CFCal_cff import *

# Reconstruction geometry services
#  Tracking Geometry
#bah - well, this is not a cfi!
from Geometry.CommonDetUnit.globalTrackingSLHCGeometry_cfi import *

#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopologyConstants_cfi import *

#Muon
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.GEMGeometry.gemGeometry_cfi import *
from Geometry.GEMGeometry.me0Geometry_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerSLHCGeometry_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *
trackerSLHCGeometry.applyAlignment = cms.bool(False)

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometryBuilder_cfi import *

CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL'          , 
                                'ZDC'           ,
                                'CASTOR'        ,
                                'EcalBarrel'    , 
                                'TOWER'           )
)

from Geometry.EcalAlgo.EcalBarrelGeometry_cfi import *
from Geometry.HcalEventSetup.HcalGeometry_cfi import *
from Geometry.HcalEventSetup.CaloTowerGeometry_cfi import *
from Geometry.HcalEventSetup.HcalTopology_cfi import *
from Geometry.ForwardGeometry.ForwardGeometry_cfi import *

from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
from Geometry.HcalCommonData.hcalRecNumberingInitialization_cfi import *

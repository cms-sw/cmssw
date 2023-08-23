import FWCore.ParameterSet.Config as cms

# Ideal geometry, needed for transient ECAL alignement
from Geometry.HcalCommonData.testPhase2GeometryFine_cff import *

#Tracker
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cff import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = False

#Muon
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.GEMGeometryBuilder.gemGeometry_cfi import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *

#  Calorimeters
from Geometry.CaloEventSetup.HGCalTopology_cfi import *
from Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometryBuilder_cfi import *
CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring('HCAL'          , 
                                'ZDC'           ,
                                'CASTOR'        ,
                                'EcalBarrel'    , 
                                "TOWER",
                                "HGCalEESensitive",
                                "HGCalHESiliconSensitive",
                                "HGCalHEScintillatorSensitive"  )
)

from Geometry.EcalAlgo.EcalBarrelGeometry_cfi import *
from Geometry.HcalEventSetup.HcalGeometry_cfi import *
from Geometry.HcalEventSetup.CaloTowerGeometry_cfi import *
from Geometry.HcalEventSetup.CaloTowerTopology_cfi import *
from Geometry.HcalCommonData.hcalDDDRecConstants_cfi import *
from Geometry.HcalEventSetup.hcalTopologyIdeal_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *

# forward
from Geometry.ForwardGeometry.ForwardGeometry_cfi import *

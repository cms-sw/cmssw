import FWCore.ParameterSet.Config as cms

# Ideal geometry, needed for transient ECAL alignement
from Configuration.Geometry.GeometryExtendedPhaseIPixel_cff import *


# Reconstruction geometry services
#  Tracking Geometry
#bah - well, this is not a cfi!
#from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
#for now - copy it in. we'll need to clone it
from Geometry.CSCGeometry.cscGeometry_cfi import *
from Geometry.RPCGeometry.rpcGeometry_cfi import *
from Geometry.DTGeometry.dtGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingSLHCGeometry_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi import *


#Tracker
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopologyConstants_cfi import *

#Muon
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *

#  Alignment
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *


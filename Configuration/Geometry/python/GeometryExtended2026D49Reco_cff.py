import FWCore.ParameterSet.Config as cms

# This config was generated automatically using generate2026Geometry.py
# If you notice a mistake, please update the generating script, not just this config

from Configuration.Geometry.GeometryExtended2026D49_cff import *

# tracker
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cfi import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = cms.bool(False)

# calo
from Geometry.CaloEventSetup.HGCalV9Topology_cfi import *
from Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometryBuilder_cfi import *
CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring("HCAL",
                                "ZDC",
                                "EcalBarrel",
                                "TOWER",
                                "HGCalEESensitive",
                                "HGCalHESiliconSensitive",
                                "HGCalHEScintillatorSensitive"
    )
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

# muon
from Geometry.MuonNumbering.muonNumberingInitialization_cfi import *
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
from Geometry.GEMGeometryBuilder.gemGeometry_cfi import *
from Geometry.GEMGeometryBuilder.me0Geometry_cfi import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *

# forward
from Geometry.ForwardGeometry.ForwardGeometry_cfi import *

# timing
from RecoMTD.DetLayers.mtdDetLayerGeometry_cfi import *
from Geometry.MTDGeometryBuilder.mtdParameters_cfi import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi import *
from Geometry.MTDNumberingBuilder.mtdTopology_cfi import *
from Geometry.MTDGeometryBuilder.mtdGeometry_cfi import *
from Geometry.MTDGeometryBuilder.idealForDigiMTDGeometry_cff import *
mtdGeometry.applyAlignment = cms.bool(False)


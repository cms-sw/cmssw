import FWCore.ParameterSet.Config as cms

# This config came from a copy of 2 files from Configuration/Geometry/python

from Configuration.Geometry.GeometryDD4hep_cff import *
DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("Geometry/HGCalCommonData/data/dd4hep/testHGCalV18.xml")

from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.HGCalCommonData.hgcalParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MuonNumbering.muonOffsetESProducer_cff import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *

# tracker
from Geometry.CommonTopologies.globalTrackingGeometry_cfi import *
from RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi import *
from Geometry.TrackerGeometryBuilder.trackerParameters_cff import *
from Geometry.TrackerNumberingBuilder.trackerTopology_cfi import *
from Geometry.TrackerGeometryBuilder.idealForDigiTrackerGeometry_cff import *
trackerGeometry.applyAlignment = False

# calo
from Geometry.CaloEventSetup.HGCalTopology_cfi import *
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
from Geometry.GEMGeometryBuilder.gemGeometry_cff import *
from Geometry.CSCGeometryBuilder.idealForDigiCscGeometry_cff import *
from Geometry.DTGeometryBuilder.idealForDigiDtGeometry_cff import *

# forward
from Geometry.ForwardGeometry.ForwardGeometry_cfi import *

# timing
from RecoMTD.DetLayers.mtdDetLayerGeometry_cfi import *
from Geometry.MTDGeometryBuilder.mtdParameters_cff import *
from Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cff import *
from Geometry.MTDNumberingBuilder.mtdTopology_cfi import *
from Geometry.MTDGeometryBuilder.mtdGeometry_cfi import *
from Geometry.MTDGeometryBuilder.idealForDigiMTDGeometry_cff import *
mtdGeometry.applyAlignment = False

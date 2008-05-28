import FWCore.ParameterSet.Config as cms

from SimGeneral.HepPDTESSource.pythiapdt_cfi import *
#The Tracker geometry ESProducer's (two producers, one for an aligned, 
# one for a misaligned geometry, identical by default
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from FastSimulation.Configuration.trackerGeometry_cfi import *
from FastSimulation.Configuration.TrackerRecoGeometryESProducer_cfi import *
from FastSimulation.TrackerSetup.TrackerInteractionGeometryESProducer_cfi import *
#The Magnetic Field ESProducer's 
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
from FastSimulation.ParticlePropagator.MagneticFieldMapESProducer_cfi import *
# The Calo geometry service model
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
# The muon geometry
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# The muon digi calibration
from CalibMuon.Configuration.DT_FakeConditions_cff import *
from CalibMuon.Configuration.CSC_FakeDBConditions_cff import *
# Services from the CondDB
from CondCore.DBCommon.CondDBSetup_cfi import *
from RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsEarlyCollision_cff import *
from RecoBTag.Configuration.RecoBTag_FakeConditions_cff import *
from RecoBTau.Configuration.RecoBTau_FakeConditions_cff import *
from CalibCalorimetry.Configuration.Hcal_FakeConditions_cff import *
from CalibCalorimetry.Configuration.Ecal_FakeConditions_cff import *
from CalibTracker.Configuration.TrackerAlignment.TrackerAlignment_Fake_cff import *
from CalibMuon.Configuration.RPC_FakeConditions_cff import *



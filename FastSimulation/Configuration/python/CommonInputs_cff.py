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
# The muon geometry
from RecoMuon.DetLayers.muonDetLayerGeometry_cfi import *
# The condDB setup
from CondCore.DBCommon.CondDBSetup_cfi import *
#Services from the CondDB (DevDB)
#include "CalibMuon/Configuration/data/DT_FrontierConditions_DevDB.cff"
#include "CalibMuon/Configuration/data/CSC_FrontierDBConditions_DevDB.cff"
#include "RecoVertex/BeamSpotProducer/data/BeamSpotEarlyCollision_DevDB.cff"
#include "RecoBTag/Configuration/data/RecoBTag_FrontierConditions_DevDB.cff"
#include "RecoBTau/Configuration/data/RecoBTau_FrontierConditions_DevDB.cff"
#include "FastSimulation/Configuration/data/Hcal_FrontierConditions_DevDB.cff"
#include "FastSimulation/Configuration/data/Ecal_FrontierConditions_DevDB.cff"
#include "FastSimulation/Configuration/data/Tracker_FrontierConditions_DevDB.cff" 
#Services from the CondDB (Int)
from CalibMuon.Configuration.DT_FrontierConditions_IntDB_cff import *
from CalibMuon.Configuration.CSC_FrontierDBConditions_IntDB_cff import *
from RecoVertex.BeamSpotProducer.BeamSpotEarlyCollision_IntDB_cff import *
from RecoBTag.Configuration.RecoBTag_FrontierConditions_IntDB_cff import *
from RecoBTau.Configuration.RecoBTau_FrontierConditions_IntDB_cff import *
from FastSimulation.Configuration.Hcal_FrontierConditions_IntDB_cff import *
from FastSimulation.Configuration.Ecal_FrontierConditions_IntDB_cff import *
from FastSimulation.Configuration.Tracker_FrontierConditions_IntDB_cff import *


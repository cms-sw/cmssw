import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.HcalCommonData.testPhase2GeometryFineXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cff import *
from SLHCUpgradeSimulations.Geometry.fakePhase2OuterTrackerConditions_cff import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.HGCalCommonData.hgcalParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi import *
from Geometry.MuonNumbering.muonGeometryConstants_cff import *
from Geometry.MuonNumbering.muonOffsetESProducer_cff import *
from Geometry.CaloEventSetup.HGCalTopology_cfi import *

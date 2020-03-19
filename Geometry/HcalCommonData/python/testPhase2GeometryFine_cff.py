import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.HcalCommonData.testPhase2GeometryFineXML_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from Geometry.EcalCommonData.ecalSimulationParameters_cff import *
from Geometry.HcalCommonData.hcalDDDSimConstants_cff import *
from Geometry.HGCalCommonData.hgcalV6ParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalV6NumberingInitialization_cfi import *
from Geometry.CaloEventSetup.HGCalTopology_cfi import *

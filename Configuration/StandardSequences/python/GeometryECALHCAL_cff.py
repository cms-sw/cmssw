import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
# for ECAL+HCAL
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.ecalhcalGeometryXML_cfi import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *






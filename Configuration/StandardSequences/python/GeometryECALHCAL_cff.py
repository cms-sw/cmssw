import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
# for ECAL+HCAL
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.ecalhcalGeometryXML_cfi import *

#L1
from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *
from RecoEcal.EgammaClusterProducers.geometryForClustering_cff import *



import FWCore.ParameterSet.Config as cms

#
# Geometry master configuration
#
# Ideal geometry, needed for simulation
from Geometry.CMSCommonData.hcalOnlyGeometryXML_cfi import *

#L1
from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
CaloGeometryBuilder.SelectedCalos = ['HCAL', 'TOWER']



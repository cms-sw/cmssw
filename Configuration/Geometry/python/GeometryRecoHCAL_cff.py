import FWCore.ParameterSet.Config as cms

#  Calorimeters
from Geometry.CaloEventSetup.CaloTopology_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.HcalCommonData.hcalRecNumberingInitialization_cfi import *
CaloGeometryBuilder.SelectedCalos = ['HCAL', 'TOWER']



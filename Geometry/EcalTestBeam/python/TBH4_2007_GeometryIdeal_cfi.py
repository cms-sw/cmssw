import FWCore.ParameterSet.Config as cms

from Geometry.EcalTestBeam.TBH4_2007_EE_XML_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *

CaloGeometryBuilder.SelectedCalos = ['EcalEndcap']

from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
from Geometry.EcalMapping.EcalMapping_cfi import *
from Geometry.EcalMapping.EcalMappingRecord_cfi import *


import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixNoPU_cfi import *
from Geometry.CaloEventSetup.CaloGeometry_cff import *
from Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi import *
# use trivial ESProducer for tests
from CalibCalorimetry.EcalTrivialCondModules.EcalTrivialCondRetrieverTB_cfi import *
from SimCalorimetry.EcalTestBeam.ecaldigi_testbeam_cfi import *
CaloGeometryBuilder.SelectedCalos = ['EcalBarrel']


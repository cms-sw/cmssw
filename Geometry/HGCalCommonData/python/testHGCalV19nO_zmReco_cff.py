import FWCore.ParameterSet.Config as cms

# This config came from a copy of 2 files from Configuration/Geometry/python

from Geometry.HGCalCommonData.testHGCalV19nO_zmXML_cfi import *
from Geometry.HGCalCommonData.hgcalColdBoxParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalColdBoxParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalColdBoxNumberingInitialization_cfi import *
from Geometry.HcalCommonData.caloSimulationParameters_cff import *

# calo
from Geometry.CaloEventSetup.HGCalTopologyColdBox_cfi import *
from Geometry.HGCalGeometry.hgcalColdBoxGeometryESProducer_cfi import *
CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring(
#                                "HCAL",
#                                "ZDC",
#                                "EcalBarrel",
#                                "TOWER",
                                "HGCalEESensitive",
#                                "HGCalHESiliconSensitive",
#                                "HGCalHEScintillatorSensitive"
    )
)

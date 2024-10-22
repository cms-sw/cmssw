import FWCore.ParameterSet.Config as cms

# This config came from a copy of 2 files from Configuration/Geometry/python

from Geometry.HGCalCommonData.testHGCalV18OXML_cfi import *
from Geometry.HGCalCommonData.hgcalParametersInitialization_cfi import *
from Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi import *
from Geometry.HcalCommonData.caloSimulationParameters_cff import *

# calo
from Geometry.CaloEventSetup.HGCalTopology_cfi import *
from Geometry.HGCalGeometry.HGCalGeometryESProducer_cfi import *
CaloGeometryBuilder = cms.ESProducer("CaloGeometryBuilder",
    SelectedCalos = cms.vstring("HCAL",
                                "ZDC",
                                "EcalBarrel",
                                "TOWER",
                                "HGCalEESensitive",
                                "HGCalHESiliconSensitive",
                                "HGCalHEScintillatorSensitive"
    )
)

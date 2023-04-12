import FWCore.ParameterSet.Config as cms

from Geometry.HGCalTBCommonData.hgcalTBEEParametersInitialize_cfi import *
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalTBEEParametersInitialize,
                fromDD4hep = True)

hgcalTBHESiParametersInitialize = hgcalTBEEParametersInitialize.clone(
    name  = "HGCalHESiliconSensitive",
    nameW = "HGCalHEWafer",
    nameC = "HGCalHECell",
    nameX = "HGCalHESiliconSensitive",
)

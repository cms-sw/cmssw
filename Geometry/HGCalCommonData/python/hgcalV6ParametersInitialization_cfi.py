import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalEEParametersInitialize,
                fromDD4hep = True )

hgcalHESiParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = "HGCalHESiliconSensitive",
    nameW = "HGCalHEWafer",
    nameC = "HGCalHECell",
    name2 = "HGCalHEsil",
    nameX = "HGCalHESiliconSensitive",
)

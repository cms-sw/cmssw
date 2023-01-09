import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalTBEEParametersInitialize_cfi import *

hgcalTBHESiParametersInitialize = hgcalTBEEParametersInitialize.clone(
    name  = "HGCalHESiliconSensitive",
    nameW = "HGCalHEWafer",
    nameC = "HGCalHECell",
    nameX = "HGCalHESiliconSensitive",
)

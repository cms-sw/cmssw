import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *

hgcalHESiParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = "HGCalHESiliconSensitive",
    nameW = "HGCalHEWafer",
    nameC = "HGCalHECell",
    nameX = "HGCalHESiliconSensitive",
)

hgcalHEScParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = "HGCalHEScintillatorSensitive",
    nameW = "HGCalWafer",
    nameC = "HGCalCell",
    nameX = "HGCalHEScintillatorSensitive",
)

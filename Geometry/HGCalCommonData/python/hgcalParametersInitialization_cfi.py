import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialization_cfi import *

hgcalHESiParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = "HGCalHESiliconLayer",
    name2 = "HGCalHESiliconSensitive",
    nameW = "HGCalHEWafer",
    nameC = "HGCalHESiliconSensitive",
    nameX = "HGCalHESiliconSensitive",
)

hgcalHEScParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = "HGCalHEScintillatorSensitive",
    name2 = "HGCalHEScintillatorSensitive",
    nameW = "HGCalWafer",
    nameC = "HGCalHEScintillatorSensitive",
    nameX = "HGCalHEScintillatorSensitive",
)

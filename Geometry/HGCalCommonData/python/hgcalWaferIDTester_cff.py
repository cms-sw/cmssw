import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalWaferIDTesterEE_cfi import *

hgcalWaferIDTesterHEF = hgcalWaferIDTesterEE.clone(
    nameSense = "HGCalHESiliconSensitive",
    fileName = "cellIDHEF.txt"
)

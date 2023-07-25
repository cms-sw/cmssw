import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalPartialIDTesterEE_cfi import *

hgcalPartialIDTesterHEF = hgcalPartialIDTesterEE.clone(
    nameDetector = "HGCalHESiliconSensitive"
)

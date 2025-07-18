import FWCore.ParameterSet.Config as cms

from Geometry.HGCalGeometry.hgcalWaferInfoEE_cfi import *

hgcalWaferInfoHE = hgcalWaferInfoEE.clone(
    detector = "HGCalHESiliconSensitive",
)

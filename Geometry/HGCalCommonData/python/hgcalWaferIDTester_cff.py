import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalWaferIDTesterHEF_cfi import *

hgcalWaferIDTesterEE = hgcalWaferIDTesterHEF.clone(
    nameSense = "HGCalEESensitive",
    fileName = "cellIDEE.txt"
)

hgcalWaferIDShiftTesterEE = hgcalWaferIDTesterHEF.clone(
    nameSense = "HGCalEESensitive",
    fileName = "cellShift.txt",
    shift = 1
)

hgcalWaferIDShiftTesterHEF = hgcalWaferIDTesterHEF.clone(
    fileName = "cellShift.txt",
    shift = 1
)

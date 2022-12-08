import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalWaferIDTesterHEF_cfi import *

hgcalWaferIDTesterEE = hgcalWaferIDTesterHEF.clone(
    nameSense = "HGCalEESensitive",
    fileName = "cellIDEE.txt"
)

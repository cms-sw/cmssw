import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *

hfnoseParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = "HGCalHFNoseSensitive",
    name2 = "HFNoseEE",
    nameW = "HFNoseWafer",
    nameC = "HFNoseCell",
    nameT = "HFNose",
    nameX = "HGCalHFNoseSensitive"
)


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

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hfnoseParametersInitialize,
                fromDD4hep = True
)

import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *


hfnoseParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHFNoseSensitive"),
    name2 = cms.string("HFNoseEE"),
    nameW = cms.string("HFNoseWafer"),
    nameC = cms.string("HFNoseCell"),
    nameT = cms.string("HFNose"),
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hfnoseParametersInitialize,
                fromDD4Hep = cms.bool(True)
)


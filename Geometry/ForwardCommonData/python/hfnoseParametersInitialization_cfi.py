import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *


hfnoseParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.untracked.string("HGCalHFNoseSensitive"),
    name2 = cms.untracked.string("HFNoseEE"),
    nameW = cms.untracked.string("HFNoseWafer"),
    nameC = cms.untracked.string("HFNoseCell"),
    nameT = cms.untracked.string("HFNose"),
)

from Configuration.Eras.Modifier_dd4hep_cff import dd4hep

dd4hep.toModify(hfnoseParametersInitialize,
                fromDD4Hep = cms.bool(True)
)


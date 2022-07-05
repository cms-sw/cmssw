import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalV15ParametersInitialization_cfi import *

hfnoseParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHFNoseSensitive"),
    name2 = cms.string("HFNoseEE"),
    nameW = cms.string("HFNoseWafer"),
    nameC = cms.string("HFNoseCell"),
    nameT = cms.string("HFNose"),
    nameX = cms.string("HGCalHFNoseSensitive"),
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hfnoseParametersInitialize,
                fromDD4hep = cms.bool(True)
)


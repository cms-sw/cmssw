import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEV15ParametersInitialization_cfi import *

hfnoseParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHFNoseSensitive"),
    name2 = cms.string("HFNoseEE"),
    nameW = cms.string("HFNoseWafer"),
    nameC = cms.string("HFNoseCell"),
    nameT = cms.string("HFNose"),
    nameX = cms.string("HGCalHFNoseSensitive"),
)

import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalEEParametersInitialize,
                fromDD4Hep = cms.bool(True)
)

hgcalHESiParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.untracked.string("HGCalHESiliconSensitive"),
    nameW = cms.untracked.string("HGCalHEWafer"),
    nameC = cms.untracked.string("HGCalHECell"),
    name2 = cms.untracked.string("HGCalHEsil"),
)

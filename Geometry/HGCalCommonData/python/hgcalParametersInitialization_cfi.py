import FWCore.ParameterSet.Config as cms

from Geometry.HGCalCommonData.hgcalEEParametersInitialize_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalEEParametersInitialize,
                fromDD4Hep = cms.bool(True)
)

hgcalHESiParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHESiliconSensitive"),
    nameW = cms.string("HGCalHEWafer"),
    nameC = cms.string("HGCalHECell"),
)

hgcalHEScParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHEScintillatorSensitive"),
    nameW = cms.string("HGCalWafer"),
    nameC = cms.string("HGCalCell"),
)

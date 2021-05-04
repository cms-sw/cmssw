import FWCore.ParameterSet.Config as cms

from CondTools.Geometry.HGCalEEParametersWriter_cfi import *

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(HGCalEEParametersWriter,
                fromDD4Hep = cms.bool(True)
)

HGCalHESiParametersWriter = HGCalEEParametersWriter.clone(
    name  = cms.string("HGCalHESiliconSensitive"),
    nameW = cms.string("HGCalHEWafer"),
    nameC = cms.string("HGCalHECell"),
)

HGCalHEScParametersWriter = HGCalEEParametersWriter.clone(
    name  = cms.string("HGCalHEScintillatorSensitive"),
    nameW = cms.string("HGCalWafer"),
    nameC = cms.string("HGCalCell"),
)

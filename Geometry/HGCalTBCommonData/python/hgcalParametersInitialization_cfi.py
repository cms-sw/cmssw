import FWCore.ParameterSet.Config as cms

hgcalTBEEParametersInitialize = cms.ESProducer('HGCalParametersESModule',
    name  = cms.string('HGCalEESensitive'),
    nameW = cms.string('HGCalEEWafer'),
    nameC = cms.string('HGCalEECell'),
    nameX = cms.string('HGCalEESensitive'),
    fromDD4hep = cms.bool(False),
    appendToDataLabel = cms.string('')
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalTBEEParametersInitialize,
                fromDD4hep = True)

hgcalTBHESiParametersInitialize = hgcalTBEEParametersInitialize.clone(
    name  = "HGCalHESiliconSensitive",
    nameW = "HGCalHEWafer",
    nameC = "HGCalHECell",
    nameX = "HGCalHESiliconSensitive",
)

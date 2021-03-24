import FWCore.ParameterSet.Config as cms

hgcalEEParametersInitialize = cms.ESProducer('HGCalParametersESModule',
  name = cms.string('HGCalEELayer'),
  name2 = cms.string('HGCalEE'),
  nameW = cms.string('HGCalEEWafer'),
  nameC = cms.string('HGCalEESensitive'),
  nameT = cms.string('HGCal'),
  nameX = cms.string('HGCalEESensitive'),
  fromDD4Hep = cms.bool(False),
  appendToDataLabel = cms.string('')
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

dd4hep.toModify(hgcalEEParametersInitialize,
                fromDD4Hep = cms.bool(True)
)

hgcalHESiParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHESiliconLayer"),
    nameW = cms.string("HGCalHEWafer"),
    nameC = cms.string("HGCalHESiliconSensitive"),
    nameX = cms.string("HGCalHESiliconSensitive"),
)

hgcalHEScParametersInitialize = hgcalEEParametersInitialize.clone(
    name  = cms.string("HGCalHEScintillatorSensitive"),
    nameW = cms.string("HGCalWafer"),
    nameC = cms.string("HGCalHEScintillatorSensitive"),
    nameX = cms.string("HGCalHEScintillatorSensitive"),
)

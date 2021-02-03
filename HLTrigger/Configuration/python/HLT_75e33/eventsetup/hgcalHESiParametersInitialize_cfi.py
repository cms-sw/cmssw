import FWCore.ParameterSet.Config as cms

hgcalHESiParametersInitialize = cms.ESProducer("HGCalParametersESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False),
    name = cms.string('HGCalHESiliconSensitive'),
    name2 = cms.string('HGCalEE'),
    nameC = cms.string('HGCalHECell'),
    nameT = cms.string('HGCal'),
    nameW = cms.string('HGCalHEWafer')
)

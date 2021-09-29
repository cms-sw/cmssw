import FWCore.ParameterSet.Config as cms

hgcalEEParametersInitialize = cms.ESProducer("HGCalParametersESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False),
    name = cms.string('HGCalEESensitive'),
    name2 = cms.string('HGCalEE'),
    nameC = cms.string('HGCalEECell'),
    nameT = cms.string('HGCal'),
    nameW = cms.string('HGCalEEWafer')
)

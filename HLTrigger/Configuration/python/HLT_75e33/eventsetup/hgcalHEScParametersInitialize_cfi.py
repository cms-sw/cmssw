import FWCore.ParameterSet.Config as cms

hgcalHEScParametersInitialize = cms.ESProducer("HGCalParametersESModule",
    appendToDataLabel = cms.string(''),
    fromDD4Hep = cms.bool(False),
    name = cms.string('HGCalHEScintillatorSensitive'),
    name2 = cms.string('HGCalEE'),
    nameC = cms.string('HGCalCell'),
    nameT = cms.string('HGCal'),
    nameW = cms.string('HGCalWafer')
)

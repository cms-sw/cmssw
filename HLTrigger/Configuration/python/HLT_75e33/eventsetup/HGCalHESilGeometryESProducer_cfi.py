import FWCore.ParameterSet.Config as cms

HGCalHESilGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
    Name = cms.untracked.string('HGCalHESiliconSensitive')
)

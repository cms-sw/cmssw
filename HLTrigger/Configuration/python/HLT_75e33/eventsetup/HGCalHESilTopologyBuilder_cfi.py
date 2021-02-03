import FWCore.ParameterSet.Config as cms

HGCalHESilTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
    Name = cms.string('HGCalHESiliconSensitive'),
    Type = cms.int32(9)
)

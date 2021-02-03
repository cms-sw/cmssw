import FWCore.ParameterSet.Config as cms

HGCalHESciTopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
    Name = cms.string('HGCalHEScintillatorSensitive'),
    Type = cms.int32(10)
)

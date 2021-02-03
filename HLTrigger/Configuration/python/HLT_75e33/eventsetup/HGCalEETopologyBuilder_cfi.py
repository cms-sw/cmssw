import FWCore.ParameterSet.Config as cms

HGCalEETopologyBuilder = cms.ESProducer("HGCalTopologyBuilder",
    Name = cms.string('HGCalEESensitive'),
    Type = cms.int32(8)
)

import FWCore.ParameterSet.Config as cms

HGCalEEGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
    Name = cms.untracked.string('HGCalEESensitive')
)

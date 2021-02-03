import FWCore.ParameterSet.Config as cms

HGCalHESciGeometryESProducer = cms.ESProducer("HGCalGeometryESProducer",
    Name = cms.untracked.string('HGCalHEScintillatorSensitive')
)

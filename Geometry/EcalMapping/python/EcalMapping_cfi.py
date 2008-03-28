import FWCore.ParameterSet.Config as cms

EcalElectronicsMappingBuilder = cms.ESProducer("EcalElectronicsMappingBuilder",
    MapFile = cms.untracked.string('Geometry/EcalMapping/data/EEMap.txt')
)



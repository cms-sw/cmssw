import FWCore.ParameterSet.Config as cms

mtdTopology = cms.ESProducer("MTDTopologyEP",
    appendToDataLabel = cms.string('')
)

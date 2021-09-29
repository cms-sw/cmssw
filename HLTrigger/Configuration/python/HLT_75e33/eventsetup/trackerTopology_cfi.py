import FWCore.ParameterSet.Config as cms

trackerTopology = cms.ESProducer("TrackerTopologyEP",
    appendToDataLabel = cms.string('')
)

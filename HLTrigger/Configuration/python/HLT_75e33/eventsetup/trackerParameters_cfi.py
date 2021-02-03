import FWCore.ParameterSet.Config as cms

trackerParameters = cms.ESProducer("TrackerParametersESModule",
    appendToDataLabel = cms.string('')
)

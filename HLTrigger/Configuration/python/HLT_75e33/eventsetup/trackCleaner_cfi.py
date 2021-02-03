import FWCore.ParameterSet.Config as cms

trackCleaner = cms.ESProducer("TrackCleanerESProducer",
    ComponentName = cms.string('trackCleaner'),
    appendToDataLabel = cms.string('')
)

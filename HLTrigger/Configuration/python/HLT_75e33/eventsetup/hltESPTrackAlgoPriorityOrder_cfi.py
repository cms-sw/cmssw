import FWCore.ParameterSet.Config as cms

hltESPTrackAlgoPriorityOrder = cms.ESProducer("TrackAlgoPriorityOrderESProducer",
    ComponentName = cms.string('hltESPTrackAlgoPriorityOrder'),
    algoOrder = cms.vstring(),
    appendToDataLabel = cms.string('')
)
# foo bar baz
# yH7Na40VTZZYh
# Hzecpdnmz6kJV

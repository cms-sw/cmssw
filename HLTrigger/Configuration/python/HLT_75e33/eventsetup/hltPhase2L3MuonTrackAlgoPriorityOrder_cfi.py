import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonTrackAlgoPriorityOrder = cms.ESProducer("TrackAlgoPriorityOrderESProducer",
    ComponentName = cms.string('hltPhase2L3MuonTrackAlgoPriorityOrder'),
    algoOrder = cms.vstring(
        'initialStep',
        'highPtTripletStep'
    ),
    appendToDataLabel = cms.string('')
)

import FWCore.ParameterSet.Config as cms

trackAlgoPriorityOrder = cms.ESProducer("TrackAlgoPriorityOrderESProducer",
    ComponentName = cms.string('trackAlgoPriorityOrder'),
    algoOrder = cms.vstring(
        'hltIter0', 
        'initialStep', 
        'highPtTripletStep'
    ),
    appendToDataLabel = cms.string('')
)

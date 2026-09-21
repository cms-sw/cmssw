import FWCore.ParameterSet.Config as cms

hltGeneralTracksTrackAlgoPriorityOrder = cms.ESProducer("TrackAlgoPriorityOrderESProducer",
    ComponentName = cms.string('hltGeneralTracksTrackAlgoPriorityOrder'),
    algoOrder = cms.vstring(
        'initialStep',
        'highPtTripletStep',
        'lowPtQuadStep',
        'lowPtTripletStep',
        'detachedQuadStep',
        'pixelPairStep',
        'muonSeededStepInOut',
        'muonSeededStepOutIn'
    ),
    appendToDataLabel = cms.string('')
)

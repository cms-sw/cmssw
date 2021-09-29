import FWCore.ParameterSet.Config as cms

trackAlgoPriorityOrder = cms.ESProducer("TrackAlgoPriorityOrderESProducer",
    ComponentName = cms.string('trackAlgoPriorityOrder'),
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

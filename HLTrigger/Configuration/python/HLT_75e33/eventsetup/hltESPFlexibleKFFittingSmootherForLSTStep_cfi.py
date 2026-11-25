import FWCore.ParameterSet.Config as cms

hltESPFlexibleKFFittingSmootherForLSTStep = cms.ESProducer("FlexibleKFFittingSmootherESProducer",
    ComponentName = cms.string('hltESPFlexibleKFFittingSmootherForLSTStep'),
    appendToDataLabel = cms.string(''),
    looperFitter = cms.string('hltESPKFFittingSmootherForLoopers'),
    standardFitter = cms.string('hltESPKFFittingSmootherForLSTStep')
)

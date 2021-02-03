import FWCore.ParameterSet.Config as cms

trackSelectionLwtnn = cms.ESProducer("LwtnnESProducer",
    ComponentName = cms.string('trackSelectionLwtnn'),
    appendToDataLabel = cms.string(''),
    fileName = cms.FileInPath('RecoTracker/FinalTrackSelectors/data/LWTNN_network_10_5_X_v1.json')
)

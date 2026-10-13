import FWCore.ParameterSet.Config as cms

hltMkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
    appendToDataLabel = cms.string('')
)

hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
    ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
    appendToDataLabel = cms.string(''),
    config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-lstStep.json'),
    backwardSearchMinPixelLayers = cms.int32(2),
    backwardSearchPromptMaxD0 = cms.double(2.0),
    maxClusterSize = cms.uint32(8),
    minPt = cms.double(0.9)
)

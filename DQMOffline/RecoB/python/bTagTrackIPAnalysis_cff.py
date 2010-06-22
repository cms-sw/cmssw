import FWCore.ParameterSet.Config as cms

# TrackIP tag info configuration
bTagTrackIPAnalysisBlock = cms.PSet(
    parameters = cms.PSet(
        endEffPur = cms.double(1.005),
        nBinEffPur = cms.int32(200),
        startEffPur = cms.double(0.005)
    )
)



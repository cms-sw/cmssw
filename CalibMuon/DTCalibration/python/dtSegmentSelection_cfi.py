import FWCore.ParameterSet.Config as cms

dtSegmentSelection = cms.PSet(
    checkNoisyChannels = cms.bool(False),
    minHitsPhi = cms.int32(7),
    minHitsZ = cms.int32(4),
    maxChi2 = cms.double(1000.0),
    maxAnglePhi = cms.double(25.),
    maxAngleZ = cms.double(999.)
)

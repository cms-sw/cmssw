import FWCore.ParameterSet.Config as cms

HiLowPtTrackingRegionWithVertexBlock = cms.PSet(
    VertexCollection = cms.InputTag("hiSelectedPixelVertex"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    fixedError = cms.double(0.2),
    nSigmaZ = cms.double(3.0),
    originRadius = cms.double(0.2),
    precise = cms.bool(True),
    ptMin = cms.double(0.25),
    sigmaZVertex = cms.double(3.0),
    useFakeVertices = cms.bool(False),
    useFixedError = cms.bool(True),
    useFoundVertices = cms.bool(True),
    useMultipleScattering = cms.bool(False)
)
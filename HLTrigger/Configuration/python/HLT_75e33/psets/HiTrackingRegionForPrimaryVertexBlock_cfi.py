import FWCore.ParameterSet.Config as cms

HiTrackingRegionForPrimaryVertexBlock = cms.PSet(
    beamSpot = cms.InputTag("offlineBeamSpot"),
    directionXCoord = cms.double(1.0),
    directionYCoord = cms.double(1.0),
    directionZCoord = cms.double(0.0),
    doVariablePtMin = cms.bool(True),
    nSigmaZ = cms.double(3.0),
    originRadius = cms.double(0.1),
    precise = cms.bool(True),
    ptMin = cms.double(0.7),
    siPixelRecHits = cms.InputTag("siPixelRecHits"),
    useFakeVertices = cms.bool(False),
    useMultipleScattering = cms.bool(False)
)
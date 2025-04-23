import FWCore.ParameterSet.Config as cms

hltTrackingRegionFromTrimmedVertices = cms.EDProducer('GlobalTrackingRegionWithVerticesEDProducer',
  RegionPSet = cms.PSet(
    ptMin                   = cms.double(0.9),
    beamSpot                = cms.InputTag('hltOnlineBeamSpot'),
    VertexCollection        = cms.InputTag('hltPhase2TrimmedPixelVertices'),
    originRadius            = cms.double(0.02),
    precise                 = cms.bool(True),
    useMultipleScattering   = cms.bool(False),
    useFixedError           = cms.bool(True),
    sigmaZVertex            = cms.double(3.0),
    fixedError              = cms.double(0.2),
    useFoundVertices        = cms.bool(True),
    useFakeVertices         = cms.bool(False),
    maxNVertices            = cms.int32(-1),
    nSigmaZ                 = cms.double(4.0),
    pixelClustersForScaling = cms.InputTag('hltSiPixelClusters'),
    originRScaling4BigEvts  = cms.bool(False),
    ptMinScaling4BigEvts    = cms.bool(False),
    halfLengthScaling4BigEvts = cms.bool(False),
    allowEmpty              = cms.bool(False),
    minOriginR              = cms.double(0),
    maxPtMin                = cms.double(1000),
    minHalfLength           = cms.double(0),
    scalingStartNPix        = cms.double(0.0),
    scalingEndNPix          = cms.double(1.0),
  )
)

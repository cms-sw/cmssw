import FWCore.ParameterSet.Config as cms

GhostTrackRecoVertexVariables = cms.PSet(
    vertexNTracks = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(21),
        max = cms.double(20.5),
        min = cms.double(-0.5)
    ),
    jetNSecondaryVertices = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(11),
        max = cms.double(10.5),
        min = cms.double(-0.5)
    )
       
)
GhostTrackPseudoVertexVariables = cms.PSet(

    flightDistance2dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(2.5),
        min = cms.double(0.0)
    ),
    flightDistance2dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        min = cms.double(0.0)
    ),
    flightDistance3dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    flightDistance3dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        min = cms.double(0.0)
    ),
    vertexJetDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.5),
        min = cms.double(0.0)
    ),
    vertexMass = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    vertexEnergyRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        min = cms.double(0.0)
    ),
    jetNSingleTrackVertices = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(11),
        max = cms.double(10.5),
        min = cms.double(-0.5)
    )

)

GhostTrackNoVertexVariables = cms.PSet(
    trackEtaRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(9.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(1.3)
    ),
    trackSip3dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-1.0)
    ),
    trackSip3dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(60.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-40.0)
    ),
    trackSip2dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(60.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-40.0)
    ),
    trackSip2dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.2),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-0.2)
    ),
    trackDecayLenVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(5.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackMomentum = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(200.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackEta = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(9.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(1.3)
    ),
    trackPtRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(15.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.3),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    vertexCategory = cms.PSet(
        logScale = cms.bool(False),
        nBins = cms.uint32(3),
        max = cms.double(2.5),
        min = cms.double(-0.5)
    ),
    trackSumJetDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.3),
        min = cms.double(0.0)
    ),
    trackSumJetEtRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(4.0),
        min = cms.double(0.0)
    ),
    trackSip3dSigAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(60.0),
        min = cms.double(-40.0)
    ),
    trackSip2dSigAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(60.0),
        min = cms.double(-40.0)
    ),
  ## New Variables used in Ghost-Track tagger
    trackGhostTrackDistVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    trackGhostTrackDistSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    trackGhostTrackWeight = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.5)
    ),
    trackChi2 = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    trackNPixelHits = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        min = cms.double(0.0)
    ),
    trackNTotalHits = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        min = cms.double(0.0)
    ),     

)

import FWCore.ParameterSet.Config as cms

combinedSVRecoVertexVariables = cms.PSet(
    flightDistance3dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.01)
    ),
    flightDistance3dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        min = cms.double(0.0)
    ),
    flightDistance2dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(2.5),
        min = cms.double(0.01)
    ),
    jetNSecondaryVertices = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    flightDistance2dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        min = cms.double(3.0)
    )
)
combinedSVPseudoVertexVariables = cms.PSet(
    vertexMass = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        min = cms.double(0.0)
    ),
    vertexNTracks = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        min = cms.double(0.0)
    ),
    vertexJetDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.5),
        min = cms.double(0.0)
    ),
    vertexEnergyRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        min = cms.double(0.0)
    )
)
combinedSVNoVertexVariables = cms.PSet(
    trackPPar = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(250.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackSip2dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.2),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-0.2)
    ),
    trackDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.3),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackSip2dSigAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        min = cms.double(-50.0)
    ),
    trackEtaRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(9.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(1.3)
    ),
    vertexCategory = cms.PSet(
        logScale = cms.bool(False),
        nBins = cms.uint32(3),
        max = cms.double(2.5),
        min = cms.double(-0.5)
    ),
    trackSip2dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-50.0)
    ),
    trackDecayLenVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(50.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackSip3dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(3.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-3.0)
    ),
    trackSumJetDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.3),
        min = cms.double(0.0)
    ),
    trackJetDist = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-10.0)
    ),
    trackSumJetEtRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(4.0),
        min = cms.double(0.0)
    ),
    trackPtRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackPtRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.5),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackMomentum = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(250.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.0)
    ),
    trackPParRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(0.86)
    ),
    trackSip3dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        indices = cms.vuint32(0, 1, 2, 3),
        min = cms.double(-50.0)
    ),
    trackSip3dSigAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(80.0),
        min = cms.double(-50.0)
    )
)



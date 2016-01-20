import FWCore.ParameterSet.Config as cms

combinedSVRecoVertexAllSoftLeptonCtagLVariables = cms.PSet(
    flightDistance2dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(50.0),
        min = cms.double(-1.0)
    ),
    flightDistance3dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(50.0),
        min = cms.double(-1.0)
    ),
    flightDistance2dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        min = cms.double(-0.1)
    ),
    flightDistance3dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(3.0),
        min = cms.double(-0.1)
    ),
    vertexFitProb = cms.PSet(
        logScale = cms.bool(False),
        nBins = cms.uint32(50),
        max = cms.double(5.0),
        min = cms.double(0.0)
    ),
    jetNSecondaryVertices = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(5),
        max = cms.double(4.5),
        min = cms.double(-0.5)
    )
)

combinedSVRecoPseudoVertexAllSoftLeptonCtagLVariables = cms.PSet(
    vertexJetDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.35),
        min = cms.double(-0.1)
    ),
    massVertexEnergyFraction = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        min = cms.double(0.0)
    ),
    vertexBoostOverSqrtJetPt = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.5),
        min = cms.double(-0.1)
    ),
    vertexNTracks = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(15),
        max = cms.double(14.5),
        min = cms.double(-0.5)
    )   
)

combinedSVPseudoVertexAllSoftLeptonCtagLVariables = cms.PSet(
    vertexMass = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(40.0),
        min = cms.double(0.0)
    ),
    vertexEnergyRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(2.0),
        min = cms.double(-0.1)
    )
)

combinedSVAllVertexAllSoftLeptonCtagLVariables = cms.PSet(
    trackSip2dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(30.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-10.0)
    ),
    trackSip3dSig = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(30.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-10.0)
    ),
    trackSip2dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.15),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-0.05)
    ),
    trackSip3dVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.15),
        indices = cms.vuint32(0,1,2),
        min = cms.double(-0.05)
    ),
    trackPtRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.0)
    ),
    trackPPar = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(200.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.0)
    ),
    trackEtaRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.0)
    ),
    trackDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.35),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.0)
    ),
    trackPtRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.35),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.0)
    ),
    trackPParRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.01),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.95)
    ),
    trackJetDist = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(100),
        max = cms.double(0.005),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-0.08)
    ),
    trackDecayLenVal = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(5.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(0.0)
    ),
    trackSip2dSigAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        min = cms.double(-5.0)
    ),
    trackSip3dSigAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(20.0),
        min = cms.double(-5.0)
    ),
    trackSumJetEtRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        min = cms.double(0.0)
    ),
    trackSumJetDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.35),
        min = cms.double(-0.05)
    ),
    jetNTracks = cms.PSet(
        logScale = cms.bool(False),
        nBins = cms.uint32(50),
        max = cms.double(60.0),
        min = cms.double(0.0)
    ),
    trackSip2dValAboveCharm  = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.15),
        min = cms.double(-0.05)
    ), 
    trackSip3dValAboveCharm = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.15),
        min = cms.double(-0.05)
    )

)

combinedSVAllVertexSoftLeptonCtagLVariables = cms.PSet(
    leptonPtRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(10.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-1.0)
    ),
    leptonSip3d = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(30.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-10.0)
    ),
    leptonDeltaR = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.4),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-0.1)
    ),
    leptonRatioRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.05),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-0.03)
    ),
    leptonEtaRel = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(0.1),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-0.1)
    ),
    leptonRatio = cms.PSet(
        logScale = cms.bool(True),
        nBins = cms.uint32(50),
        max = cms.double(1.0),
        indices = cms.vuint32(0, 1, 2),
        min = cms.double(-0.1)
    )
)


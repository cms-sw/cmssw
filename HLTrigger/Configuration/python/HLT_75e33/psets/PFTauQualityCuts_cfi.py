import FWCore.ParameterSet.Config as cms

PFTauQualityCuts = cms.PSet(
    isolationQualityCuts = cms.PSet(
        maxDeltaZ = cms.double(0.15),
        maxTrackChi2 = cms.double(100.0),
        maxTransverseImpactParameter = cms.double(0.05),
        minGammaEt = cms.double(1.5),
        minTrackHits = cms.uint32(8),
        minTrackPixelHits = cms.uint32(0),
        minTrackPt = cms.double(1.0),
        minTrackVertexWeight = cms.double(-1.0)
    ),
    leadingTrkOrPFCandOption = cms.string('leadPFCand'),
    primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
    pvFindingAlgo = cms.string('closestInDeltaZ'),
    recoverLeadingTrk = cms.bool(False),
    signalQualityCuts = cms.PSet(
        maxDeltaZ = cms.double(0.4),
        maxTrackChi2 = cms.double(100.0),
        maxTransverseImpactParameter = cms.double(0.1),
        minGammaEt = cms.double(1.0),
        minNeutralHadronEt = cms.double(30.0),
        minTrackHits = cms.uint32(3),
        minTrackPixelHits = cms.uint32(0),
        minTrackPt = cms.double(0.5),
        minTrackVertexWeight = cms.double(-1.0)
    ),
    vertexTrackFiltering = cms.bool(False),
    vxAssocQualityCuts = cms.PSet(
        maxTrackChi2 = cms.double(100.0),
        maxTransverseImpactParameter = cms.double(0.1),
        minGammaEt = cms.double(1.0),
        minTrackHits = cms.uint32(3),
        minTrackPixelHits = cms.uint32(0),
        minTrackPt = cms.double(0.5),
        minTrackVertexWeight = cms.double(-1.0)
    )
)
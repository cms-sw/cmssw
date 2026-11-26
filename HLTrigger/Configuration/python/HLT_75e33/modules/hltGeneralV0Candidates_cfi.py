import FWCore.ParameterSet.Config as cms

# K0s -> pi+ pi- and Lambda -> p pi- reconstruction from HLT tracks,
# adapted from RecoVertex/V0Producer/python/generalV0Candidates_cfi.py
# Produces reco::VertexCompositeCandidateCollection with instances "Kshort" and "Lambda"
hltGeneralV0Candidates = cms.EDProducer("V0Producer",
    # which beamSpot to reference
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),

    # reference primary vertex instead of beamSpot
    useVertex = cms.bool(False),
    # which vertex collection to use
    vertices = cms.InputTag("hltOfflinePrimaryVertices"),

    # which TrackCollection to use for vertexing
    trackRecoAlgorithm = cms.InputTag("hltGeneralTracks"),

    # which V0s to reconstruct
    doKShorts = cms.bool(True),
    doLambdas = cms.bool(True),

    # perform vertex fit (if false, the POCA is used in lieu of the fitted point)
    doFit = cms.bool(True),

    # True -> KalmanVertexFitter, False -> AdaptiveVertexFitter
    vertexFitter = cms.bool(True),

    # use the refitted tracks returned from the KVF for V0Candidate kinematics
    useRefTracks = cms.bool(True),

    # -- cuts on initial track collection --
    tkChi2Cut = cms.double(10.),
    tkNHitsCut = cms.int32(3),
    tkPtCut = cms.double(0.35),
    tkIPSigXYCut = cms.double(2.),
    tkIPSigZCut = cms.double(-1.),

    # -- cuts on the vertex --
    vtxChi2Cut = cms.double(6.63),
    vtxDecaySigXYCut = cms.double(15.),
    vtxDecayXYCut = cms.double(-1.),
    ssVtxDecayXYCut = cms.double(-1.),
    vtxDecaySigXYZCut = cms.double(-1.),

    # -- miscellaneous cuts --
    allowSS = cms.bool(False),
    innerOuterTkDCAThreshold = cms.double(5.),
    innerTkDCACut = cms.double(1.),
    outerTkDCACut = cms.double(1.),
    allowWideAngleVtx = cms.bool(False),
    mPiPiCut = cms.double(0.6),
    # requires TrackExtras (kept by hltGeneralTracks via copyExtras); set to -1 to disable
    innerHitPosCut = cms.double(4.),
    cosThetaXYCut = cms.double(0.998),
    cosThetaXYZCut = cms.double(-2.),

    # -- cuts on the V0 candidate mass --
    kShortMassCut = cms.double(0.07),
    lambdaMassCut = cms.double(0.05)
)

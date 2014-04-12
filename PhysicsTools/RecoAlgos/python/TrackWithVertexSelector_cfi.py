import FWCore.ParameterSet.Config as cms

trackWithVertexSelector = cms.EDFilter("TrackWithVertexSelector",
    # the track collection
    src = cms.InputTag("ctfWithMaterialTracks"),
    d0Max = cms.double(10.0),
    # quality cuts (valid hits, normalized chi2)
    numberOfValidHits = cms.uint32(8),
    normalizedChi2 = cms.double(999999.0),
    copyExtras = cms.untracked.bool(False), ## copies also extras and rechits on RECO

    nVertices = cms.uint32(3), ## how many vertices to look at before dropping the track

    vtxFallback = cms.bool(True), ## falback to beam spot if there are no vertices

    # don't set this to true on AOD!
    copyTrajectories = cms.untracked.bool(False),
    # uses vtx=(0,0,0) with deltaZeta=15.9, deltaRho = 0.2
    zetaVtx = cms.double(1.0),
    numberOfLostHits = cms.uint32(999), ## at most 999 lost hits

    etaMin = cms.double(0.0),
    vertexTag = cms.InputTag("offlinePrimaryVerticesFromCTFTracks"),
    # kinematic cuts  (pT in GeV)
    ptMin = cms.double(0.0),
    numberOfValidPixelHits = cms.uint32(0), ## at least <n> hits inthe pixeles

    rhoVtx = cms.double(0.2), ## tags used by b-tagging folks

    # impact parameter cuts (in cm)
    dzMax = cms.double(35.0),
    ptMax = cms.double(14000.0),
    etaMax = cms.double(5.0),
    # compatibility with a vertex ?
    useVtx = cms.bool(True)
)



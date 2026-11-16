import FWCore.ParameterSet.Config as cms

# Run-3 hltL3fL1sMu9HP10IP6L3Filtered0 (HLT_GRun), adapted to the Phase-2 L3 muons:
# - CandTag/inputMuonCollection: Phase-2 L3 muons, BeamSpotTag: hltOnlineBeamSpot;
# - MatchToPreviousCand = False: the Run-3 L3 -> L2 -> L1 matching (InputLinks, L1CandTag) does
#   not apply to Phase-2, the L1 matching is done by hltL3fL1TkSingleMuBarrelL3Filtered9.
# The other selections are the Run-3 ones: pT > 9 GeV, |eta| < 2.5, |dxy|/sigma(dxy) > 6 with
# respect to the beam spot, at least 2 matched muon stations.
hltL3fL1TkSingleMuBarrelIP6L3Filtered9 = cms.EDFilter("HLTMuonL3PreFilter",
    saveTags = cms.bool(True),
    BeamSpotTag = cms.InputTag("hltOnlineBeamSpot"),
    CandTag = cms.InputTag("hltPhase2L3MuonCandidates"),
    PreviousCandTag = cms.InputTag("hltL3fL1TkSingleMuBarrelL3Filtered9"),
    L1CandTag = cms.InputTag(""),
    inputMuonCollection = cms.InputTag("hltPhase2L3Muons"),
    MinN = cms.int32(1),
    MaxEta = cms.double(2.5),
    MinNhits = cms.int32(0),
    MaxDr = cms.double(2.0),
    MinDr = cms.double(-1.0),
    MaxDz = cms.double(9999.0),
    MinDxySig = cms.double(6.0),
    MinPt = cms.double(9.0),
    NSigmaPt = cms.double(0.0),
    MaxNormalizedChi2 = cms.double(9999.0),
    MaxDXYBeamSpot = cms.double(9999.0),
    MinDXYBeamSpot = cms.double(-1.0),
    MinNmuonHits = cms.int32(0),
    MaxPtDifference = cms.double(9999.0),
    MinTrackPt = cms.double(0.0),
    minMuonStations = cms.int32(2),
    minTrkHits = cms.int32(-1),
    minMuonHits = cms.int32(-1),
    allowedTypeMask = cms.uint32(255),
    requiredTypeMask = cms.uint32(0),
    MaxNormalizedChi2_L3FromL1 = cms.double(1.0E99),
    trkMuonId = cms.uint32(0),
    L1MatchingdR = cms.double(0.3),
    MatchToPreviousCand = cms.bool(False),
    InputLinks = cms.InputTag(""),
    requireL3MuonTrajectorySeed = cms.bool(False),
    useSimpleGeometry = cms.bool(True),
    useStation2 = cms.bool(True),
    fallbackToME1 = cms.bool(False),
    cosmicPropagationHypothesis = cms.bool(False),
    useMB2InOverlap = cms.bool(False),
    useTrack = cms.string("tracker"),
    useState = cms.string("atVertex"),
    propagatorAlong = cms.ESInputTag("", "hltESPSteppingHelixPropagatorAlong"),
    propagatorAny = cms.ESInputTag("", "SteppingHelixPropagatorAny"),
    propagatorOpposite = cms.ESInputTag("", "hltESPSteppingHelixPropagatorOpposite")
)

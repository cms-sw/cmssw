import FWCore.ParameterSet.Config as cms

hltPhase2L3FromL1TkMuonPixelVertices = cms.EDProducer("PixelVertexProducer",
    Finder = cms.string('DivisiveVertexFinder'),
    Method2 = cms.bool(True),
    NTrkMin = cms.int32(2),
    PVcomparer = cms.PSet(
        refToPSet_ = cms.string('hltPhase2PSetPvClusterComparerForIT')
    ),
    PtMin = cms.double(1.0),
    TrackCollection = cms.InputTag("hltPhase2L3FromL1TkMuonPixelTracks"),
    UseError = cms.bool(True),
    Verbosity = cms.int32(0),
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5.0),
    ZSeparation = cms.double(0.05),
    beamSpot = cms.InputTag("offlineBeamSpot")
)

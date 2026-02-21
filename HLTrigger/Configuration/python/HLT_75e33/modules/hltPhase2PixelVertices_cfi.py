import FWCore.ParameterSet.Config as cms

hltPhase2PixelVertices = cms.EDProducer("PixelVertexProducer",
    Finder = cms.string('DivisiveVertexFinder'),
    Method2 = cms.bool(True),
    NTrkMin = cms.int32(2),
    PVcomparer = cms.PSet(
        refToPSet_ = cms.string('pSetPvClusterComparerForIT')
    ),
    PtMin = cms.double(1.0),
    # Even though pixel tracks with a highPurity ID, i.e. hltPhase2PixelTracks,
    # are used in other tracking modules, the pixel tracks without an ID,
    # i.e. hltPhase2PixelTracksCAExtension, are used here.
    # This avoids a circular dependency, as the highPurity ID requires a vertex,
    # while also providing satisfactory physics performance.
    # To be improved with a DNN-based highPurity ID that does not depend on vertices.
    TrackCollection = cms.InputTag("hltPhase2PixelTracksCAExtension"),
    UseError = cms.bool(True),
    Verbosity = cms.int32(0),
    WtAverage = cms.bool(True),
    ZOffset = cms.double(5.0),
    ZSeparation = cms.double(0.005),
    beamSpot = cms.InputTag("hltOnlineBeamSpot")
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
hltPhase2LegacyTracking.toModify(hltPhase2PixelVertices,
    TrackCollection = "hltPhase2PixelTracks"
)

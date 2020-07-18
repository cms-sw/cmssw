import FWCore.ParameterSet.Config as cms

hltPhase2PixelVertices = cms.EDProducer(
    "PixelVertexProducer",
    Finder=cms.string("DivisiveVertexFinder"),
    Method2=cms.bool(True),
    NTrkMin=cms.int32(2),
    PVcomparer=cms.PSet(refToPSet_=cms.string("hltPhase2PSetPvClusterComparerForIT")),
    PtMin=cms.double(1.0),
    TrackCollection=cms.InputTag("hltPhase2PixelTracks"),
    UseError=cms.bool(True),
    Verbosity=cms.int32(0),
    WtAverage=cms.bool(True),
    ZOffset=cms.double(5.0),
    ZSeparation=cms.double(0.05),
    beamSpot=cms.InputTag("offlineBeamSpot"),
)

# process.hltPhase2TrimmedPixelVertices = cms.EDProducer(
#    "PixelVertexCollectionTrimmer",
#    PVcomparer=cms.PSet(refToPSet_=cms.string("hltPhase2PSetPvClusterComparerForIT")),
#    fractionSumPt2=cms.double(0.3),
#    maxVtx=cms.uint32(100),
#    minSumPt2=cms.double(0.0),
#    src=cms.InputTag("hltPhase2PixelVertices"),
# )

hltPhase2PixelVerticesSequence = cms.Sequence(
    hltPhase2PixelVertices  # + hltPhase2TrimmedPixelVertices
)

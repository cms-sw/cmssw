import FWCore.ParameterSet.Config as cms

hltIter0Phase2L3FromL1TkMuonPixelSeedsFromPixelTracks = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag("hltPhase2L3FromL1TkMuonPixelTracks"),
    InputVertexCollection = cms.InputTag("hltPhase2L3FromL1TkMuonTrimmedPixelVertices"),
    SeedCreatorPSet = cms.PSet(
        refToPSet_ = cms.string('hltPhase2SeedFromProtoTracks')
    ),
    originHalfLength = cms.double(0.3),
    originRadius = cms.double(0.1),
    useProtoTrackKinematics = cms.bool(False),
    sortAndFilterProtoTracks = cms.bool(False),
    useEventsWithNoVertex = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    usePV = cms.bool(False),
    includeFourthHit = cms.bool(False),
    removeOTRechits = cms.bool(False),
    produceComplement = cms.bool(False)
)

import FWCore.ParameterSet.Config as cms

hltInitialStepSeeds = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag("hltPhase2PixelTracks"),
    InputVertexCollection = cms.InputTag(""),
    SeedCreatorPSet = cms.PSet(
        refToPSet_ = cms.string('seedFromProtoTracks')
    ),
    originHalfLength = cms.double(0.3),
    originRadius = cms.double(0.1),
    useProtoTrackKinematics = cms.bool(False),
    sortAndFilterProtoTracks = cms.bool(False),
    useEventsWithNoVertex = cms.bool(True),
    TTRHBuilder = cms.string('WithTrackAngle'),
    usePV = cms.bool(False),
    includeFourthHit = cms.bool(True),
    removeOTRechits = cms.bool(False),
    produceComplement = cms.bool(False)
)

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
hltPhase2LegacyTracking.toModify(hltInitialStepSeeds,
    includeFourthHit = False
)

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
phase2_hlt_vertexTrimming.toModify(hltInitialStepSeeds, InputVertexCollection = "hltPhase2TrimmedPixelVertices")

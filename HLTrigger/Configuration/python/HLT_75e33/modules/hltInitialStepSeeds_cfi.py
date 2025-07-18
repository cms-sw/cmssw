import FWCore.ParameterSet.Config as cms

hltInitialStepSeeds = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag("hltPhase2PixelTracks"),
    InputVertexCollection = cms.InputTag(""),
    SeedCreatorPSet = cms.PSet(
        refToPSet_ = cms.string('seedFromProtoTracks')
    ),
    TTRHBuilder = cms.string('WithTrackAngle'),
    originHalfLength = cms.double(0.3),
    originRadius = cms.double(0.1),
    useEventsWithNoVertex = cms.bool(True),
    usePV = cms.bool(False),
    useProtoTrackKinematics = cms.bool(False),
    includeFourthHit = cms.bool(False)
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toModify(hltInitialStepSeeds, includeFourthHit = True)

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
phase2_hlt_vertexTrimming.toModify(hltInitialStepSeeds, InputVertexCollection = "hltPhase2TrimmedPixelVertices")

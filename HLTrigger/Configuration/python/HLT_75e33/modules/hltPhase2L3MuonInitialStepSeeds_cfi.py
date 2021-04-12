import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonInitialStepSeeds = cms.EDProducer("SeedGeneratorFromProtoTracksEDProducer",
    InputCollection = cms.InputTag("hltPhase2L3MuonPixelTracks"),
    InputVertexCollection = cms.InputTag(""),
    SeedCreatorPSet = cms.PSet(
        refToPSet_ = cms.string('hltPhase2L3MuonSeedFromProtoTracks')
    ),
    TTRHBuilder = cms.string('WithTrackAngle'),
    originHalfLength = cms.double(0.3),
    originRadius = cms.double(0.1),
    useEventsWithNoVertex = cms.bool(True),
    usePV = cms.bool(True),
    useProtoTrackKinematics = cms.bool(False)
)

import FWCore.ParameterSet.Config as cms

pfTrack = cms.EDProducer("PFTrackProducer",
    GsfTrackModuleLabel = cms.InputTag("electronGsfTracks"),
    GsfTracksInEvents = cms.bool(False),
    MuColl = cms.InputTag("hltPhase2L3Muons"),
    PrimaryVertexLabel = cms.InputTag("offlinePrimaryVertices"),
    TkColList = cms.VInputTag(cms.InputTag("generalTracks")),
    TrackQuality = cms.string('highPurity'),
    TrajInEvents = cms.bool(False),
    UseQuality = cms.bool(True)
)

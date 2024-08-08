import FWCore.ParameterSet.Config as cms

hltPfTrack = cms.EDProducer("PFTrackProducer",
    GsfTrackModuleLabel = cms.InputTag("electronGsfTracks"),
    GsfTracksInEvents = cms.bool(False),
    MuColl = cms.InputTag("hltPhase2L3Muons"),
    PrimaryVertexLabel = cms.InputTag("hltOfflinePrimaryVertices"),
    TkColList = cms.VInputTag(cms.InputTag("hltGeneralTracks")),
    TrackQuality = cms.string('highPurity'),
    TrajInEvents = cms.bool(False),
    UseQuality = cms.bool(True)
)

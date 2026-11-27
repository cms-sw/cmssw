import FWCore.ParameterSet.Config as cms

l1tTrackerEtMiss = cms.EDProducer("L1TrackerEtMissProducer",
    L1MetCollectionName = cms.string('L1TrackerEtMiss'),
    L1TrackAssociatedInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForEtMiss","Level1TTTracksSelectedAssociated"),
    L1TrackInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss","Level1TTTracksSelected"),
    debug = cms.bool(False),
    highPtTracks = cms.int32(1),
    maxPt = cms.double(-10.0)
)

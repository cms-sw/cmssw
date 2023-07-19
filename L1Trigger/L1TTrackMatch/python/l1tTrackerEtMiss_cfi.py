import FWCore.ParameterSet.Config as cms

l1tTrackerEtMiss = cms.EDProducer('L1TrackerEtMissProducer',
    L1TrackInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelected"),
    L1TrackAssociatedInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForEtMiss", "Level1TTTracksSelectedAssociated"),
    L1MetCollectionName = cms.string("L1TrackerEtMiss"),
    maxPt = cms.double(-10.) ,	    # in GeV. When maxPt > 0, tracks with PT above maxPt are considered as
                                    # mismeasured and are treated according to highPtTracks below.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    highPtTracks = cms.int32(1) ,   # when = 0 : truncation. Tracks with PT above maxPt are ignored
                                    # when = 1 : saturation. Tracks with PT above maxPt are set to PT=maxPt.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    debug     = cms.bool(False)
)

l1tTrackerEtMissExtended = l1tTrackerEtMiss.clone( #NOT OPTIMIZED, STUDIED, OR USED
    L1TrackInputTag = ("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelected"),
    L1TrackAssociatedInputTag = ("l1tTrackVertexAssociationProducerExtendedForEtMiss", "Level1TTTracksExtendedSelectedAssociated"),
    L1MetCollectionName = "L1TrackerExtendedEtMiss",
)

import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer
from L1Trigger.L1TTrackMatch.L1TrackSelectionProducer_cfi import L1TrackSelectionProducer, L1TrackSelectionProducerExtended

L1TrackerEtMiss = cms.EDProducer('L1TrackerEtMissProducer',
    L1TrackInputTag = cms.InputTag("L1TrackSelectionProducer", L1TrackSelectionProducer.outputCollectionName.value()),
    L1TrackAssociatedInputTag = cms.InputTag("L1TrackSelectionProducer", L1TrackSelectionProducer.outputCollectionName.value() + "Associated"),
    L1MetCollectionName = cms.string("L1TrackerEtMiss"),
    maxPt = cms.double( -10. ),	    # in GeV. When maxPt > 0, tracks with PT above maxPt are considered as
                                    # mismeasured and are treated according to highPtTracks below.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    highPtTracks = cms.int32( 1 ),  # when = 0 : truncation. Tracks with PT above maxPt are ignored
                                    # when = 1 : saturation. Tracks with PT above maxPt are set to PT=maxPt.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    debug     = cms.bool(False)
)

L1TrackerEtMissExtended = L1TrackerEtMiss.clone( #NOT OPTIMIZED, STUDIED, OR USED
    L1TrackInputTag = cms.InputTag("L1TrackSelectionProducerExtended", L1TrackSelectionProducerExtended.outputCollectionName.value()),
    L1TrackAssociatedInputTag = cms.InputTag("L1TrackSelectionProducerExtended", L1TrackSelectionProducerExtended.outputCollectionName.value() + "Associated"),
    L1MetCollectionName = cms.string("L1TrackerExtendedEtMiss"),
)

import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import l1tVertexProducer
from L1Trigger.L1TTrackMatch.L1TrackSelectionProducer_cfi import l1tTrackSelectionProducer, l1tTrackSelectionProducerExtended

l1tTrackerEtMiss = cms.EDProducer('L1TrackerEtMissProducer',
    L1TrackInputTag = cms.InputTag("L1TrackSelectionProducer", l1tTrackSelectionProducer.outputCollectionName.value()),
    L1TrackAssociatedInputTag = cms.InputTag("L1TrackSelectionProducer", l1tTrackSelectionProducer.outputCollectionName.value() + "Associated"),
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
    L1TrackInputTag = ("L1TrackSelectionProducerExtended", l1tTrackSelectionProducerExtended.outputCollectionName.value()),
    L1TrackAssociatedInputTag = ("L1TrackSelectionProducerExtended", l1tTrackSelectionProducerExtended.outputCollectionName.value() + "Associated"),
    L1MetCollectionName = "L1TrackerExtendedEtMiss",
)

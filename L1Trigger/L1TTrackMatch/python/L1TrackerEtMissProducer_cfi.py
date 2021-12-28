import FWCore.ParameterSet.Config as cms
from L1Trigger.VertexFinder.VertexProducer_cff import VertexProducer
from L1Trigger.L1TTrackMatch.L1TrackSelectionProducer_cfi import L1TrackSelectionProducer, L1TrackSelectionProducerExtended

L1TrackerEtMiss = cms.EDProducer('L1TrackerEtMissProducer',
    L1TrackInputTag = cms.InputTag("L1TrackSelectionProducer", L1TrackSelectionProducer.outputCollectionName.value()),
    L1VertexInputTag = cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()),
    L1MetCollectionName = cms.string("L1TrackerEtMiss"),
    deltaZ = cms.double( 3. ),      # in cm
    maxPt = cms.double( 200. ),	    # in GeV. When maxPt > 0, tracks with PT above maxPt are considered as
                                    # mismeasured and are treated according to highPtTracks below.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    highPtTracks = cms.int32( 1 ),  # when = 0 : truncation. Tracks with PT above maxPt are ignored
                                    # when = 1 : saturation. Tracks with PT above maxPt are set to PT=maxPt.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    displaced = cms.bool(False),     # Use promt/displaced tracks
    z0Thresholds = cms.vdouble( 0.37, 0.5, 0.6, 0.75, 1.0, 1.6 ), # Threshold for track to vertex association.
    etaRegions = cms.vdouble( 0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4 ), # Eta bins for choosing deltaZ threshold.
    debug     = cms.bool(False)
)

L1TrackerEtMissExtended = cms.EDProducer('L1TrackerEtMissProducer', #NOT OPTIMIZED, STUDIED, OR USED
    L1TrackInputTag = cms.InputTag("L1TrackSelectionProducerExtended", L1TrackSelectionProducerExtended.outputCollectionName.value()),
    L1VertexInputTag = cms.InputTag("VertexProducer", VertexProducer.l1VertexCollectionName.value()),
    L1MetCollectionName = cms.string("L1TrackerEtMiss"),
    L1MetExtendedCollectionName = cms.string("L1TrackerExtendedEtMiss"),
    deltaZ = cms.double( 3.0 ),     # in cm
    maxPt = cms.double( 200. ),	    # in GeV. When maxPt > 0, tracks with PT above maxPt are considered as
                                    # mismeasured and are treated according to highPtTracks below.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    highPtTracks = cms.int32( 1 ),  # when = 0 : truncation. Tracks with PT above maxPt are ignored
                                    # when = 1 : saturation. Tracks with PT above maxPt are set to PT=maxPt.
                                    # When maxPt < 0, no special treatment is done for high PT tracks.
    displaced = cms.bool(True),      # Use promt/displaced tracks
    z0Thresholds = cms.vdouble( 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 ), # Threshold for track to vertex association.
    etaRegions = cms.vdouble( 0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4 ), # Eta bins for choosing deltaZ threshold.
    debug     = cms.bool(False)
)

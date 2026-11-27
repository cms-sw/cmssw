import FWCore.ParameterSet.Config as cms

l1tTrackerEmuEtMiss = cms.EDProducer("L1TrackerEtMissEmulatorProducer",
    L1MetCollectionName = cms.string('L1TrackerEmuEtMiss'),
    L1TrackAssociatedInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForEtMiss","Level1TTTracksSelectedAssociatedEmulation"),
    L1TrackInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss","Level1TTTracksSelectedEmulation"),
    L1VertexInputTag = cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation"),
    debug = cms.int32(0),
    nCordicSteps = cms.int32(13),
    useGTTinput = cms.bool(True)
)

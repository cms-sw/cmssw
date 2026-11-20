import FWCore.ParameterSet.Config as cms

l1tTrackVertexAssociationProducerExtendedForEtMiss = cms.EDProducer("L1TrackVertexAssociationProducer",
    cutSet = cms.PSet(
        deltaZMax = cms.vdouble(
            3.0, 3.0, 3.0, 3.0, 3.0,
            3.0
        ),
        deltaZMaxEtaBounds = cms.vdouble(
            0.0, 0.7, 1.0, 1.2, 1.6,
            2.0, 2.4
        )
    ),
    debug = cms.int32(0),
    fwNTrackSetsTVA = cms.uint32(94),
    l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss","Level1TTTracksExtendedSelectedEmulation"),
    l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss","Level1TTTracksExtendedSelected"),
    l1TracksInputTag = cms.InputTag("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),
    l1VerticesEmulationInputTag = cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation"),
    l1VerticesInputTag = cms.InputTag("l1tVertexFinder","L1Vertices"),
    outputCollectionName = cms.string('Level1TTTracksExtendedSelectedAssociated'),
    processEmulatedTracks = cms.bool(True),
    processSimulatedTracks = cms.bool(True),
    useDisplacedTracksDeltaZOverride = cms.double(3.0)
)

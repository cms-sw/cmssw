import FWCore.ParameterSet.Config as cms

l1tTrackVertexAssociationProducerForJets = cms.EDProducer("L1TrackVertexAssociationProducer",
    cutSet = cms.PSet(
        deltaZMax = cms.vdouble(0.55),
        deltaZMaxEtaBounds = cms.vdouble(0.0, 2.4)
    ),
    debug = cms.int32(0),
    fwNTrackSetsTVA = cms.uint32(94),
    l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerForJets","Level1TTTracksSelectedEmulation"),
    l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerForJets","Level1TTTracksSelected"),
    l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),
    l1VerticesEmulationInputTag = cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation"),
    l1VerticesInputTag = cms.InputTag("l1tVertexFinder","L1Vertices"),
    outputCollectionName = cms.string('Level1TTTracksSelectedAssociated'),
    processEmulatedTracks = cms.bool(True),
    processSimulatedTracks = cms.bool(True),
    useDisplacedTracksDeltaZOverride = cms.double(-1.0)
)

import FWCore.ParameterSet.Config as cms

l1tTrackVertexAssociationProducer = cms.EDProducer('L1TrackVertexAssociationProducer',
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducer", "Level1TTTracksSelectedEmulation"),
  # If no vertex collection is provided, then the DeltaZ cuts will not be run
  l1VerticesInputTag = cms.InputTag("l1tVertexFinder", "l1vertices"),
  l1VerticesEmulationInputTag = cms.InputTag("l1tVertexFinderEmulator", "l1verticesEmulation"),
  outputCollectionName = cms.string("Level1TTTracksSelectedAssociated"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(0.37, 0.50, 0.60, 0.75, 1.00, 1.60), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = cms.double(-1.0), # override the deltaZ cut value for displaced tracks
  processSimulatedTracks = cms.bool(True), # return selected tracks after cutting on the floating point values
  processEmulatedTracks = cms.bool(True), # return selected tracks after cutting on the bitwise emulated values
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3, 4
)

l1tTrackVertexAssociationProducerExtended = l1tTrackVertexAssociationProducer.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtended", "Level1TTTracksExtendedSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtended", "Level1TTTracksExtendedSelectedEmulation"),
  outputCollectionName = cms.string("Level1TTTracksExtendedSelectedAssociated"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(3.0, 3.0, 3.0, 3.0, 3.0, 3.0), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = cms.double(3.0), # Use promt/displaced tracks
)

l1tTrackVertexAssociationProducerForJets = l1tTrackVertexAssociationProducer.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerForJets", "Level1TTTracksSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerForJets", "Level1TTTracksSelectedEmulation"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
)

l1tTrackVertexAssociationProducerExtendedForJets = l1tTrackVertexAssociationProducerExtended.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForJets", "Level1TTTracksExtendedSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForJets", "Level1TTTracksExtendedSelectedEmulation"),
  cutSet = cms.PSet(
                    #deltaZMaxEtaBounds = cms.vdouble(0.0, absEtaMax.value), # these values define the bin boundaries in |eta|
                    #deltaZMax = cms.vdouble(0.5), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(999.0, 999.0, 999.0, 999.0, 999.0, 999.0), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
)

l1tTrackVertexAssociationProducerForEtMiss = l1tTrackVertexAssociationProducer.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerForEtMiss", "Level1TTTracksSelectedEmulation"),
)

l1tTrackVertexAssociationProducerExtendedForEtMiss = l1tTrackVertexAssociationProducerExtended.clone(
  l1SelectedTracksInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelected"),
  l1SelectedTracksEmulationInputTag = cms.InputTag("l1tTrackSelectionProducerExtendedForEtMiss", "Level1TTTracksExtendedSelectedEmulation"),
)



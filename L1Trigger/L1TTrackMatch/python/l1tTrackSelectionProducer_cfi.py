import FWCore.ParameterSet.Config as cms

l1tTrackSelectionProducer = cms.EDProducer('L1TrackSelectionProducer',
  l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),
  # If no vertex collection is provided, then the DeltaZ cuts will not be run
  l1VerticesInputTag = cms.InputTag("l1tVertexFinder", "l1vertices"),
  l1VerticesEmulationInputTag = cms.InputTag("l1tVertexFinderEmulator", "l1verticesEmulation"),
  outputCollectionName = cms.string("Level1TTTracksSelected"),
  cutSet = cms.PSet(
                    ptMin = cms.double(2.0), # pt must be greater than this value, [GeV]
                    absEtaMax = cms.double(2.4), # absolute value of eta must be less than this value
                    absZ0Max = cms.double(15.0), # z0 must be less than this value, [cm]
                    nStubsMin = cms.int32(4), # number of stubs must be greater than or equal to this value
                    nPSStubsMin = cms.int32(0), # the number of stubs in the PS Modules must be greater than or equal to this value

                    reducedBendChi2Max = cms.double(2.25), # bend chi2 must be less than this value
                    reducedChi2RZMax = cms.double(5.0), # chi2rz/dof must be less than this value
                    reducedChi2RPhiMax = cms.double(20.0), # chi2rphi/dof must be less than this value

                    deltaZMaxEtaBounds = cms.vdouble(0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4), # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble(0.37, 0.50, 0.60, 0.75, 1.00, 1.60), # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = cms.double(-1.0), # override the deltaZ cut value for displaced tracks
  processSimulatedTracks = cms.bool(True), # return selected tracks after cutting on the floating point values
  processEmulatedTracks = cms.bool(True), # return selected tracks after cutting on the bitwise emulated values
  debug = cms.int32(0) # Verbosity levels: 0, 1, 2, 3, 4
)

l1tTrackSelectionProducerExtended = l1tTrackSelectionProducer.clone(
  l1TracksInputTag = ("l1tGTTInputProducerExtended","Level1TTTracksExtendedConverted"),
  outputCollectionName = "Level1TTTracksExtendedSelected",
  cutSet = dict(
                    ptMin = 3.0, # pt must be greater than this value, [GeV]
                    absEtaMax = 2.4, # absolute value of eta must be less than this value
                    absZ0Max = 15.0, # z0 must be less than this value, [cm]
                    nStubsMin = 4, # number of stubs must be greater than or equal to this value
                    nPSStubsMin = 0, # the number of stubs in the PS Modules must be greater than or equal to this value

                    reducedBendChi2Max = 2.4, # bend chi2 must be less than this value
                    reducedChi2RZMax = 10.0, # chi2rz/dof must be less than this value
                    reducedChi2RPhiMax = 40.0, # chi2rphi/dof must be less than this value

                    deltaZMaxEtaBounds = [0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4], # these values define the bin boundaries in |eta|
                    deltaZMax = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0], # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = 3.0, # Use prompt/displaced tracks
)



import FWCore.ParameterSet.Config as cms

L1TrackSelectionProducer = cms.EDProducer('L1TrackSelectionProducer',
  l1TracksInputTag = cms.InputTag("L1GTTInputProducer","Level1TTTracksConverted"),
  # If no vertex collection is provided, then the DeltaZ cuts will not be run
  l1VerticesInputTag = cms.InputTag("L1VertexFinder", "l1vertices"),
  l1VerticesEmulationInputTag = cms.InputTag("L1VertexFinderEmulator", "l1verticesEmulation"),
  outputCollectionName = cms.string("Level1TTTracksSelected"),
  cutSet = cms.PSet(
                    ptMin = 2.0, # pt must be greater than this value, [GeV]
                    absEtaMax = 2.4, # absolute value of eta must be less than this value
                    absZ0Max = 15.0, # z0 must be less than this value, [cm]
                    nStubsMin = 4, # number of stubs must be greater than or equal to this value
                    nPSStubsMin = 0, # the number of stubs in the PS Modules must be greater than or equal to this value

                    reducedBendChi2Max = 2.25, # bend chi2 must be less than this value
                    reducedChi2RZMax = 5.0, # chi2rz/dof must be less than this value
                    reducedChi2RPhiMax = 20.0, # chi2rphi/dof must be less than this value

                    deltaZMaxEtaBounds = [0.0, 0.7, 1.0, 1.2, 1.6, 2.0, 2.4], # these values define the bin boundaries in |eta|
                    deltaZMax = cms.vdouble[0.37, 0.50, 0.60, 0.75, 1.00, 1.60], # delta z must be less than these values, there will be one less value here than in deltaZMaxEtaBounds, [cm]
                    ),
  useDisplacedTracksDeltaZOverride = -1.0, # override the deltaZ cut value for displaced tracks
  processSimulatedTracks = True, # return selected tracks after cutting on the floating point values
  processEmulatedTracks = True, # return selected tracks after cutting on the bitwise emulated values
  debug = 0 # Verbosity levels: 0, 1, 2, 3, 4
)

L1TrackSelectionProducerExtended = L1TrackSelectionProducer.clone(
  l1TracksInputTag = ("L1GTTInputProducerExtended","Level1TTTracksExtendedConverted"),
  outputCollectionName = "Level1TTTracksExtendedSelected",
  cutSet = cms.PSet(
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
  useDisplacedTracksDeltaZOverride = 3.0, # Use promt/displaced tracks
)



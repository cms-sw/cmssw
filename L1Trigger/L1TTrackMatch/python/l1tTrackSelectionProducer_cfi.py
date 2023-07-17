import FWCore.ParameterSet.Config as cms

l1tTrackSelectionProducer = cms.EDProducer('L1TrackSelectionProducer',
  l1TracksInputTag = cms.InputTag("l1tGTTInputProducer","Level1TTTracksConverted"),
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
                    ),
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
                    ),
  processSimulatedTracks = cms.bool(True), # return selected tracks after cutting on the floating point values
  processEmulatedTracks = cms.bool(True), # return selected tracks after cutting on the bitwise emulated values
)

l1tTrackSelectionProducerForJets = l1tTrackSelectionProducer.clone(
  cutSet = dict(
                    ptMin = 0.0, # pt must be greater than this value, [GeV]
                    absEtaMax = 999.9, # absolute value of eta must be less than this value
                    absZ0Max = 999.9, # z0 must be less than this value, [cm]
                    nStubsMin = 0, # number of stubs must be greater than or equal to this value
                    nPSStubsMin = 0, # the number of stubs in the PS Modules must be greater than or equal to this value

                    reducedBendChi2Max = 999.9, # bend chi2 must be less than this value
                    reducedChi2RZMax = 999.9, # chi2rz/dof must be less than this value
                    reducedChi2RPhiMax = 999.9, # chi2rphi/dof must be less than this value
                    ),
)

l1tTrackSelectionProducerExtendedForJets = l1tTrackSelectionProducerExtended.clone(
  cutSet = dict(
                    ptMin = 0.0, # pt must be greater than this value, [GeV]
                    absEtaMax = 999.9, # absolute value of eta must be less than this value
                    absZ0Max = 999.9, # z0 must be less than this value, [cm]
                    nStubsMin = 0, # number of stubs must be greater than or equal to this value
                    nPSStubsMin = 0, # the number of stubs in the PS Modules must be greater than or equal to this value

                    reducedBendChi2Max = 999.9, # bend chi2 must be less than this value
                    reducedChi2RZMax = 999.9, # chi2rz/dof must be less than this value
                    reducedChi2RPhiMax = 999.9, # chi2rphi/dof must be less than this value
                    ),
)

l1tTrackSelectionProducerForEtMiss = l1tTrackSelectionProducer.clone()

l1tTrackSelectionProducerExtendedForEtMiss = l1tTrackSelectionProducerExtended.clone()



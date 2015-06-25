import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.packedCandidateTrackValidator_cfi import *

packedCandidateTrackValidatorLostTracks = packedCandidateTrackValidator.clone(
    trackToPackedCandiadteAssociation = "lostTracks",
    rootFolder = "Tracking/PackedCandidate/lostTracks"
)

tracksDQMMiniAOD = cms.Sequence(
    packedCandidateTrackValidator +
    packedCandidateTrackValidatorLostTracks
)

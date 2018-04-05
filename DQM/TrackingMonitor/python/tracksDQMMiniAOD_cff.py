import FWCore.ParameterSet.Config as cms

from DQM.TrackingMonitor.packedCandidateTrackValidator_cfi import *

packedCandidateTrackValidatorLostTracks = packedCandidateTrackValidator.clone(
    trackToPackedCandidateAssociation = "lostTracks",
    rootFolder = "Tracking/PackedCandidate/lostTracks"
)

tracksDQMMiniAOD = cms.Sequence(
    packedCandidateTrackValidator +
    packedCandidateTrackValidatorLostTracks
)

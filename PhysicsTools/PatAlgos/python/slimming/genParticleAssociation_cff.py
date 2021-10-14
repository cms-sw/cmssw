import FWCore.ParameterSet.Config as cms
import SimTracker.TrackAssociation.packedCandidatesGenAssociationDefault_cfi as _mod

packedPFCandidateToGenAssociation = _mod.packedCandidatesGenAssociationDefault.clone(
    trackToGenAssoc = "prunedTrackMCMatch",
)

lostTracksToGenAssociation = _mod.packedCandidatesGenAssociationDefault.clone(
    trackToGenAssoc = "prunedTrackMCMatch",
    trackToPackedCandidatesAssoc = "lostTracks"
)

packedCandidateToGenAssociationTask = cms.Task(packedPFCandidateToGenAssociation,lostTracksToGenAssociation)

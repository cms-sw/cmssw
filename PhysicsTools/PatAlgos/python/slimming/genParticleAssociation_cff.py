import FWCore.ParameterSet.Config as cms

packedPFCandidateToGenAssociation = cms.EDProducer("PackedCandidateGenAssociationProducer",
    trackToGenAssoc = cms.InputTag("prunedTrackMCMatch"),
)

lostTracksToGenAssociation = cms.EDProducer("PackedCandidateGenAssociationProducer",
    trackToGenAssoc = cms.InputTag("prunedTrackMCMatch"),
    trackToPackedCandidatesAssoc = cms.InputTag("lostTracks")
)

packedCandidateToGenAssociationTask = cms.Task(packedPFCandidateToGenAssociation,lostTracksToGenAssociation)

import FWCore.ParameterSet.Config as cms

l1tTrackFastJets = cms.EDProducer("L1TrackFastJetProducer",
    L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForJets","Level1TTTracksSelectedAssociated"),
    coneSize = cms.double(0.4),
    displaced = cms.bool(False)
)

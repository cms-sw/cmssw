import FWCore.ParameterSet.Config as cms

l1tTrackFastJets = cms.EDProducer("L1TrackFastJetProducer",
    L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForJets", "Level1TTTracksSelectedAssociated"),
    coneSize=cms.double(0.4),         #cone size for anti-kt fast jet
    displaced = cms.bool(False)       # use prompt/displaced tracks
)

l1tTrackFastJetsExtended = cms.EDProducer("L1TrackFastJetProducer",
    L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets", "Level1TTTracksExtendedSelectedAssociated"),
    coneSize=cms.double(0.4),         #cone size for anti-kt fast jet
    displaced = cms.bool(True)        # use prompt/displaced tracks
)

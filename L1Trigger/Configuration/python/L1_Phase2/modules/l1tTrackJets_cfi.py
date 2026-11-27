import FWCore.ParameterSet.Config as cms

l1tTrackJets = cms.EDProducer("L1TrackJetProducer",
    L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerForJets","Level1TTTracksSelectedAssociated"),
    d0_cutNStubs4 = cms.double(-1),
    d0_cutNStubs5 = cms.double(-1),
    displaced = cms.bool(False),
    etaBins = cms.int32(24),
    highpTJetMinTrackMultiplicity = cms.int32(3),
    highpTJetThreshold = cms.double(100.0),
    lowpTJetMinTrackMultiplicity = cms.int32(2),
    lowpTJetThreshold = cms.double(50.0),
    minTrkJetpT = cms.double(-1.0),
    nDisplacedTracks = cms.int32(2),
    phiBins = cms.int32(27),
    trk_etaMax = cms.double(2.4),
    trk_ptMax = cms.double(200.0),
    trk_zMax = cms.double(15.0),
    zBins = cms.int32(1)
)

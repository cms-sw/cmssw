import FWCore.ParameterSet.Config as cms

l1tTrackJetsExtendedEmulation = cms.EDProducer("L1TrackJetEmulatorProducer",
    L1TrackInputTag = cms.InputTag("l1tTrackVertexAssociationProducerExtendedForJets","Level1TTTracksExtendedSelectedAssociatedEmulation"),
    d0_cutNStubs4 = cms.double(-1),
    d0_cutNStubs5 = cms.double(0.22),
    displaced = cms.bool(True),
    etaBins = cms.int32(24),
    export_binmap = cms.bool(False),
    highpTJetMinTrackMultiplicity = cms.int32(3),
    highpTJetThreshold = cms.double(100.0),
    lowpTJetMinTrackMultiplicity = cms.int32(2),
    lowpTJetThreshold = cms.double(50.0),
    minTrkJetTrackMultiplicity = cms.int32(1),
    minTrkJetpT = cms.double(5.0),
    nDisplacedTracks = cms.int32(3),
    phiBins = cms.int32(27),
    trk_etaMax = cms.double(2.4),
    trk_ptMax = cms.double(200.0),
    trk_zMax = cms.double(15.0),
    zBins = cms.int32(1)
)

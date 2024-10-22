import FWCore.ParameterSet.Config as cms

hltHpsPFTauPrimaryVertexProducerForDeepTau = cms.EDProducer("PFTauPrimaryVertexProducer",
    Algorithm = cms.int32(0),
    ElectronTag = cms.InputTag("hltEgammaCandidates"),
    MuonTag = cms.InputTag("hltMuons"),
    PFTauTag = cms.InputTag("hltHpsPFTauProducer"),
    PVTag = cms.InputTag("hltPhase2PixelVertices"),
    RemoveElectronTracks = cms.bool(False),
    RemoveMuonTracks = cms.bool(False),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    cut = cms.string('pt > 18.0 & abs(eta)<2.4'),
    discriminators = cms.VPSet(cms.PSet(
        discriminator = cms.InputTag("hltHpsPFTauDiscriminationByDecayModeFindingNewDMs"),
        selectionCut = cms.double(0.5)
    )),
    qualityCuts = cms.PSet(
        isolationQualityCuts = cms.PSet(
            maxDeltaZ = cms.double(0.2),
            maxDeltaZToLeadTrack = cms.double(-1.0),
            maxTrackChi2 = cms.double(100.0),
            maxTransverseImpactParameter = cms.double(0.03),
            minGammaEt = cms.double(1.5),
            minTrackHits = cms.uint32(8),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(1.0),
            minTrackVertexWeight = cms.double(-1.0)
        ),
        leadingTrkOrPFCandOption = cms.string('leadPFCand'),
        primaryVertexSrc = cms.InputTag("hltPhase2PixelVertices"),
        pvFindingAlgo = cms.string('closestInDeltaZ'),
        recoverLeadingTrk = cms.bool(False),
        signalQualityCuts = cms.PSet(
            maxDeltaZ = cms.double(0.4),
            maxDeltaZToLeadTrack = cms.double(-1.0),
            maxTrackChi2 = cms.double(100.0),
            maxTransverseImpactParameter = cms.double(0.1),
            minGammaEt = cms.double(1.0),
            minNeutralHadronEt = cms.double(30.0),
            minTrackHits = cms.uint32(3),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(0.5),
            minTrackVertexWeight = cms.double(-1.0)
        ),
        vertexTrackFiltering = cms.bool(False),
        vxAssocQualityCuts = cms.PSet(
            maxTrackChi2 = cms.double(100.0),
            maxTransverseImpactParameter = cms.double(0.1),
            minGammaEt = cms.double(1.0),
            minTrackHits = cms.uint32(3),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(0.5),
            minTrackVertexWeight = cms.double(-1.0)
        )
    ),
    useBeamSpot = cms.bool(True),
    useSelectedTaus = cms.bool(False)
)

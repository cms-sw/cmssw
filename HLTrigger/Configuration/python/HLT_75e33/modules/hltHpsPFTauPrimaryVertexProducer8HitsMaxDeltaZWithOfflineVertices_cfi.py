import FWCore.ParameterSet.Config as cms

hltHpsPFTauPrimaryVertexProducer8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFTauPrimaryVertexProducer",
    Algorithm = cms.int32(0),
    ElectronTag = cms.InputTag(""),
    MuonTag = cms.InputTag(""),
    PFTauTag = cms.InputTag("hltSelectedHpsPFTaus8HitsMaxDeltaZWithOfflineVertices"),
    PVTag = cms.InputTag("offlinePrimaryVertices"),
    RemoveElectronTracks = cms.bool(False),
    RemoveMuonTracks = cms.bool(False),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    cut = cms.string('pt > 20.0 & abs(eta) < 2.4'),
    discriminators = cms.VPSet(cms.PSet(
        discriminator = cms.InputTag("hltHpsPFTauDiscriminationByDecayModeFindingNewDMs8HitsMaxDeltaZWithOfflineVertices"),
        selectionCut = cms.double(0.5)
    )),
    mightGet = cms.optional.untracked.vstring,
    qualityCuts = cms.PSet(
        isolationQualityCuts = cms.PSet(
            maxDeltaZ = cms.double(0.15),
            maxDeltaZToLeadTrack = cms.double(-1.0),
            maxTrackChi2 = cms.double(100.0),
            maxTransverseImpactParameter = cms.double(0.05),
            minGammaEt = cms.double(1.5),
            minTrackHits = cms.uint32(8),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(1.0),
            minTrackVertexWeight = cms.double(-1.0)
        ),
        leadingTrkOrPFCandOption = cms.string('leadPFCand'),
        primaryVertexSrc = cms.InputTag("offlinePrimaryVertices"),
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

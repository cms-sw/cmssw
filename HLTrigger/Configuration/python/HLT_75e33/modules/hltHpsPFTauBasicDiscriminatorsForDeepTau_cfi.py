import FWCore.ParameterSet.Config as cms

hltHpsPFTauBasicDiscriminatorsForDeepTau = cms.EDProducer("PFRecoTauDiscriminationByIsolationContainer",
    IDWPdefinitions = cms.VPSet(),
    IDdefinitions = cms.VPSet(
        cms.PSet(
            ApplyDiscriminationByTrackerIsolation = cms.bool(True),
            IDname = cms.string('ChargedIsoPtSum'),
            storeRawSumPt = cms.bool(True)
        ),
        cms.PSet(
            ApplyDiscriminationByECALIsolation = cms.bool(True),
            IDname = cms.string('NeutralIsoPtSum'),
            storeRawSumPt = cms.bool(True)
        ),
        cms.PSet(
            ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
            IDname = cms.string('NeutralIsoPtSumWeight'),
            UseAllPFCandsForWeights = cms.bool(True),
            storeRawSumPt = cms.bool(True)
        ),
        cms.PSet(
            IDname = cms.string('TauFootprintCorrection'),
            storeRawFootprintCorrection = cms.bool(True)
        ),
        cms.PSet(
            IDname = cms.string('PhotonPtSumOutsideSignalCone'),
            storeRawPhotonSumPt_outsideSignalCone = cms.bool(True)
        ),
        cms.PSet(
            IDname = cms.string('PUcorrPtSum'),
            applyDeltaBetaCorrection = cms.bool(True),
            storeRawPUsumPt = cms.bool(True)
        )
    ),
    PFTauProducer = cms.InputTag("hltHpsPFTauProducer"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('and')
    ),
    WeightECALIsolation = cms.double(1.0),
    applyFootprintCorrection = cms.bool(False),
    applyRhoCorrection = cms.bool(False),
    customOuterCone = cms.double(0.5),
    deltaBetaFactor = cms.string('0.2000'),
    deltaBetaPUTrackPtCutOverride = cms.bool(True),
    deltaBetaPUTrackPtCutOverride_val = cms.double(0.5),
    footprintCorrections = cms.VPSet(
        cms.PSet(
            offset = cms.string('0.0'),
            selection = cms.string('decayMode() = 0')
        ),
        cms.PSet(
            offset = cms.string('0.0'),
            selection = cms.string('decayMode() = 1 || decayMode() = 2')
        ),
        cms.PSet(
            offset = cms.string('2.7'),
            selection = cms.string('decayMode() = 5')
        ),
        cms.PSet(
            offset = cms.string('0.0'),
            selection = cms.string('decayMode() = 6')
        ),
        cms.PSet(
            offset = cms.string('max(2.0, 0.22*pt() - 2.0)'),
            selection = cms.string('decayMode() = 10')
        )
    ),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    minTauPtForNoIso = cms.double(-99.0),
    particleFlowSrc = cms.InputTag("hltParticleFlowTmp"),
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
            minTrackVertexWeight = cms.double(-1.0),
            useTracksInsteadOfPFHadrons = cms.bool(False)
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
            minTrackVertexWeight = cms.double(-1.0),
            useTracksInsteadOfPFHadrons = cms.bool(False)
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
    rhoConeSize = cms.double(0.5),
    rhoProducer = cms.InputTag("hltFixedGridRhoFastjetAll"),
    rhoUEOffsetCorrection = cms.double(1.0),
    verbosity = cms.int32(0),
    vertexSrc = cms.InputTag("hltPhase2PixelVertices")
)

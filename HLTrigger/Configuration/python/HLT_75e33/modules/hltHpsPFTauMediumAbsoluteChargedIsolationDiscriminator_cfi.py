import FWCore.ParameterSet.Config as cms

hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminator = cms.EDProducer("PFRecoTauDiscriminationByIsolation",
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(False),
    PFTauProducer = cms.InputTag("hltHpsPFTauProducer"),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string('and')
    ),
    UseAllPFCandsForWeights = cms.bool(False),
    WeightECALIsolation = cms.double(0.33333),
    applyDeltaBetaCorrection = cms.bool(False),
    applyFootprintCorrection = cms.bool(False),
    applyOccupancyCut = cms.bool(False),
    applyPhotonPtSumOutsideSignalConeCut = cms.bool(False),
    applyRelativeSumPtCut = cms.bool(False),
    applyRhoCorrection = cms.bool(False),
    applySumPtCut = cms.bool(True),
    customOuterCone = cms.double(-1.0),
    deltaBetaFactor = cms.string('0.38'),
    deltaBetaPUTrackPtCutOverride = cms.bool(True),
    deltaBetaPUTrackPtCutOverride_val = cms.double(0.5),
    enableHGCalWorkaround = cms.bool(False),
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
    isoConeSizeForDeltaBeta = cms.double(0.3),
    maxAbsPhotonSumPt_outsideSignalCone = cms.double(1000000000.0),
    maxRelPhotonSumPt_outsideSignalCone = cms.double(0.1),
    maximumOccupancy = cms.uint32(0),
    maximumSumPtCut = cms.double(3.7),
    minTauPtForNoIso = cms.double(-99.0),
    particleFlowSrc = cms.InputTag("hltParticleFlowTmp"),
    qualityCuts = cms.PSet(
        isolationQualityCuts = cms.PSet(
            maxDeltaZ = cms.double(0.2),
            maxTrackChi2 = cms.double(100.0),
            maxTransverseImpactParameter = cms.double(0.1),
            minGammaEt = cms.double(0.5),
            minTrackHits = cms.uint32(3),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(0.5),
            useTracksInsteadOfPFHadrons = cms.bool(False)
        ),
        primaryVertexSrc = cms.InputTag("hltPhase2PixelVertices"),
        pvFindingAlgo = cms.string('closestInDeltaZ'),
        recoverLeadingTrk = cms.bool(False),
        signalQualityCuts = cms.PSet(
            maxDeltaZ = cms.double(0.2),
            maxTrackChi2 = cms.double(1000.0),
            maxTransverseImpactParameter = cms.double(0.2),
            minGammaEt = cms.double(0.5),
            minNeutralHadronEt = cms.double(1.0),
            minTrackHits = cms.uint32(3),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(0.0),
            useTracksInsteadOfPFHadrons = cms.bool(False)
        ),
        vertexTrackFiltering = cms.bool(False),
        vxAssocQualityCuts = cms.PSet(
            maxTrackChi2 = cms.double(1000.0),
            maxTransverseImpactParameter = cms.double(0.2),
            minGammaEt = cms.double(0.5),
            minTrackHits = cms.uint32(3),
            minTrackPixelHits = cms.uint32(0),
            minTrackPt = cms.double(0.0),
            useTracksInsteadOfPFHadrons = cms.bool(False)
        )
    ),
    relativeSumPtCut = cms.double(0.03),
    relativeSumPtOffset = cms.double(0.0),
    rhoConeSize = cms.double(0.357),
    rhoProducer = cms.InputTag("NotUsed"),
    rhoUEOffsetCorrection = cms.double(0.0),
    storeRawFootprintCorrection = cms.bool(False),
    storeRawOccupancy = cms.bool(False),
    storeRawPUsumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    verbosity = cms.int32(0),
    vertexSrc = cms.InputTag("NotUsed")
)

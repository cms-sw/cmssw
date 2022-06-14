import FWCore.ParameterSet.Config as cms

hltHpsPFTauCombinatoricProducer8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("RecoTauProducer",
    buildNullTaus = cms.bool(False),
    builders = cms.VPSet(cms.PSet(
        decayModes = cms.VPSet(
            cms.PSet(
                maxPiZeros = cms.uint32(0),
                maxTracks = cms.uint32(6),
                nCharged = cms.uint32(1),
                nPiZeros = cms.uint32(0)
            ),
            cms.PSet(
                maxPiZeros = cms.uint32(6),
                maxTracks = cms.uint32(6),
                nCharged = cms.uint32(1),
                nPiZeros = cms.uint32(1)
            ),
            cms.PSet(
                maxPiZeros = cms.uint32(5),
                maxTracks = cms.uint32(6),
                nCharged = cms.uint32(1),
                nPiZeros = cms.uint32(2)
            ),
            cms.PSet(
                maxPiZeros = cms.uint32(0),
                maxTracks = cms.uint32(6),
                nCharged = cms.uint32(3),
                nPiZeros = cms.uint32(0)
            ),
            cms.PSet(
                maxPiZeros = cms.uint32(3),
                maxTracks = cms.uint32(6),
                nCharged = cms.uint32(3),
                nPiZeros = cms.uint32(1)
            )
        ),
        isolationConeSize = cms.double(0.5),
        minAbsPhotonSumPt_insideSignalCone = cms.double(2.5),
        minAbsPhotonSumPt_outsideSignalCone = cms.double(1000000000.0),
        minRelPhotonSumPt_insideSignalCone = cms.double(0.1),
        minRelPhotonSumPt_outsideSignalCone = cms.double(1000000000.0),
        name = cms.string('combinatoric'),
        pfCandSrc = cms.InputTag("particleFlowTmp"),
        plugin = cms.string('RecoTauBuilderCombinatoricPlugin'),
        qualityCuts = cms.PSet(
            isolationQualityCuts = cms.PSet(
                maxDeltaZ = cms.double(0.15),
                maxDeltaZToLeadTrack = cms.double(-1.0),
                maxTrackChi2 = cms.double(100.0),
                maxTransverseImpactParameter = cms.double(0.05),
                minGammaEt = cms.double(1.5),
                minTrackHits = cms.uint32(8),
                minTrackPixelHits = cms.uint32(0),
                minTrackPt = cms.double(0.9),
                minTrackVertexWeight = cms.double(-1.0)
            ),
            leadingTrkOrPFCandOption = cms.string('minLeadTrackOrPFCand'),
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
                minTrackPt = cms.double(0.9),
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
        signalConeSize = cms.string('min(max(3.6/pt(), 0.08), 0.12)'),
        verbosity = cms.int32(0)
    )),
    chargedHadronSrc = cms.InputTag("hltHpsPFTauPFJetsRecoTauChargedHadrons8HitsMaxDeltaZWithOfflineVertices"),
    jetRegionSrc = cms.InputTag("hltHpsPFTauPFJets08Region8HitsMaxDeltaZWithOfflineVertices"),
    jetSrc = cms.InputTag("hltHpsPFTauAK4PFJets8HitsMaxDeltaZWithOfflineVertices"),
    maxJetAbsEta = cms.double(4.0),
    minJetPt = cms.double(14.0),
    modifiers = cms.VPSet(cms.PSet(
        name = cms.string('tau_mass'),
        plugin = cms.string('PFRecoTauMassPlugin'),
        verbosity = cms.int32(0)
    )),
    outputSelection = cms.string('leadChargedHadrCand().isNonnull()'),
    piZeroSrc = cms.InputTag("hltHpsPFTauPiZeros8HitsMaxDeltaZWithOfflineVertices")
)

import FWCore.ParameterSet.Config as cms

hltHpsPFTauPiZeros8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("RecoTauPiZeroProducer",
    builders = cms.VPSet(cms.PSet(
        applyElecTrackQcuts = cms.bool(False),
        makeCombinatoricStrips = cms.bool(False),
        maxStripBuildIterations = cms.int32(-1),
        minGammaEtStripAdd = cms.double(1.0),
        minGammaEtStripSeed = cms.double(1.0),
        minStripEt = cms.double(1.0),
        name = cms.string('s'),
        plugin = cms.string('RecoTauPiZeroStripPlugin2'),
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
        stripCandidatesParticleIds = cms.vint32(2, 4),
        stripEtaAssociationDistance = cms.double(0.05),
        stripPhiAssociationDistance = cms.double(0.2),
        updateStripAfterEachDaughter = cms.bool(False),
        verbosity = cms.int32(0)
    )),
    jetSrc = cms.InputTag("hltHpsPFTauAK4PFJets8HitsMaxDeltaZWithOfflineVertices"),
    massHypothesis = cms.double(0.136),
    maxJetAbsEta = cms.double(4.0),
    mightGet = cms.optional.untracked.vstring,
    minJetPt = cms.double(14.0),
    outputSelection = cms.string('pt > 0'),
    ranking = cms.VPSet(cms.PSet(
        name = cms.string('InStrip'),
        plugin = cms.string('RecoTauPiZeroStringQuality'),
        selection = cms.string('algoIs("kStrips")'),
        selectionFailValue = cms.double(1000),
        selectionPassFunction = cms.string('abs(mass() - 0.13579)')
    )),
    verbosity = cms.int32(0)
)

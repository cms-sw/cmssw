import FWCore.ParameterSet.Config as cms

hltPFTauPiZeros = cms.EDProducer("RecoTauPiZeroProducer",
    builders = cms.VPSet(cms.PSet(
        applyElecTrackQcuts = cms.bool(False),
        makeCombinatoricStrips = cms.bool(False),
        maxStripBuildIterations = cms.int32(-1),
        minGammaEtStripAdd = cms.double(0.0),
        minGammaEtStripSeed = cms.double(0.5),
        minStripEt = cms.double(1.0),
        name = cms.string('s'),
        plugin = cms.string('RecoTauPiZeroStripPlugin2'),
        qualityCuts = cms.PSet(
            primaryVertexSrc = cms.InputTag("hltPhase2PixelVertices"),
            pvFindingAlgo = cms.string('closestInDeltaZ'),
            recoverLeadingTrk = cms.bool(False),
            signalQualityCuts = cms.PSet(
                maxDeltaZ = cms.double(0.2),
                maxTrackChi2 = cms.double(1000.0),
                maxTransverseImpactParameter = cms.double(0.2),
                minGammaEt = cms.double(0.5),
                minTrackHits = cms.uint32(3),
                minTrackPixelHits = cms.uint32(0),
                minTrackPt = cms.double(0.0),
                useTracksInsteadOfPFHadrons = cms.bool(False)
            ),
            vertexTrackFiltering = cms.bool(False)
        ),
        stripCandidatesParticleIds = cms.vint32(2, 4),
        stripEtaAssociationDistance = cms.double(0.05),
        stripPhiAssociationDistance = cms.double(0.2),
        updateStripAfterEachDaughter = cms.bool(False)
    )),
    jetSrc = cms.InputTag("hltAK4PFJets"),
    massHypothesis = cms.double(0.136),
    maxJetAbsEta = cms.double(99.0),
    minJetPt = cms.double(-1.0),
    outputSelection = cms.string('pt > 0'),
    ranking = cms.VPSet(cms.PSet(
        name = cms.string('InStrip'),
        plugin = cms.string('RecoTauPiZeroStringQuality'),
        selection = cms.string("algoIs(\'kStrips\')"),
        selectionFailValue = cms.double(1000.0),
        selectionPassFunction = cms.string('abs(mass() - 0.13579)')
    )),
    verbosity = cms.int32(0)
)

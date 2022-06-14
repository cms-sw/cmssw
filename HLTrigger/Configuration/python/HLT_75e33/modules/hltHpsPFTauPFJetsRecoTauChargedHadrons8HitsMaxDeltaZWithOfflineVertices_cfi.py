import FWCore.ParameterSet.Config as cms

hltHpsPFTauPFJetsRecoTauChargedHadrons8HitsMaxDeltaZWithOfflineVertices = cms.EDProducer("PFRecoTauChargedHadronProducer",
    builders = cms.VPSet(cms.PSet(
        chargedHadronCandidatesParticleIds = cms.vint32(1, 2, 3),
        dRmergeNeutralHadronWrtChargedHadron = cms.double(0.005),
        dRmergeNeutralHadronWrtElectron = cms.double(0.05),
        dRmergeNeutralHadronWrtNeutralHadron = cms.double(0.01),
        dRmergeNeutralHadronWrtOther = cms.double(0.005),
        dRmergePhotonWrtChargedHadron = cms.double(0.005),
        dRmergePhotonWrtElectron = cms.double(0.005),
        dRmergePhotonWrtNeutralHadron = cms.double(0.01),
        dRmergePhotonWrtOther = cms.double(0.005),
        maxUnmatchedBlockElementsNeutralHadron = cms.int32(1),
        maxUnmatchedBlockElementsPhoton = cms.int32(1),
        minBlockElementMatchesNeutralHadron = cms.int32(2),
        minBlockElementMatchesPhoton = cms.int32(2),
        minMergeChargedHadronPt = cms.double(100.0),
        minMergeGammaEt = cms.double(1.0),
        minMergeNeutralHadronEt = cms.double(1.0),
        name = cms.string('chargedPFCandidates'),
        plugin = cms.string('PFRecoTauChargedHadronFromPFCandidatePlugin'),
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
        verbosity = cms.int32(0)
    )),
    jetSrc = cms.InputTag("hltHpsPFTauAK4PFJets8HitsMaxDeltaZWithOfflineVertices"),
    maxJetAbsEta = cms.double(4.0),
    mightGet = cms.optional.untracked.vstring,
    minJetPt = cms.double(14.0),
    outputSelection = cms.string('pt > 0.90'),
    ranking = cms.VPSet(cms.PSet(
        name = cms.string('ChargedPFCandidate'),
        plugin = cms.string('PFRecoTauChargedHadronQualityPluginHGCalWorkaround')
    )),
    verbosity = cms.int32(0)
)

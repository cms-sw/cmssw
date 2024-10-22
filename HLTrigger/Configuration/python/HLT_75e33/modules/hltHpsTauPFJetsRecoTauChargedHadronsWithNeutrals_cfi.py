import FWCore.ParameterSet.Config as cms

hltHpsTauPFJetsRecoTauChargedHadronsWithNeutrals = cms.EDProducer("PFRecoTauChargedHadronProducer",
    builders = cms.VPSet(
        cms.PSet(
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
            minMergeGammaEt = cms.double(0.0),
            minMergeNeutralHadronEt = cms.double(0.0),
            name = cms.string('chargedPFCandidates'),
            plugin = cms.string('PFRecoTauChargedHadronFromPFCandidatePlugin'),
            qualityCuts = cms.PSet(
                primaryVertexSrc = cms.InputTag("hltPhase2PixelVertices"),
                pvFindingAlgo = cms.string('closestInDeltaZ'),
                recoverLeadingTrk = cms.bool(False),
                signalQualityCuts = cms.PSet(
                    maxDeltaZ = cms.double(0.2),
                    maxTrackChi2 = cms.double(1000.0),
                    maxTransverseImpactParameter = cms.double(0.2),
                    minGammaEt = cms.double(0.5),
                    minNeutralHadronEt = cms.double(30.0),
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
            )
        ),
        cms.PSet(
            chargedHadronCandidatesParticleIds = cms.vint32(5),
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
            minMergeChargedHadronPt = cms.double(0.0),
            minMergeGammaEt = cms.double(0.0),
            minMergeNeutralHadronEt = cms.double(0.0),
            name = cms.string('PFNeutralHadrons'),
            plugin = cms.string('PFRecoTauChargedHadronFromPFCandidatePlugin'),
            qualityCuts = cms.PSet(
                primaryVertexSrc = cms.InputTag("hltPhase2PixelVertices"),
                pvFindingAlgo = cms.string('closestInDeltaZ'),
                recoverLeadingTrk = cms.bool(False),
                signalQualityCuts = cms.PSet(
                    maxDeltaZ = cms.double(0.2),
                    maxTrackChi2 = cms.double(1000.0),
                    maxTransverseImpactParameter = cms.double(0.2),
                    minGammaEt = cms.double(0.5),
                    minNeutralHadronEt = cms.double(30.0),
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
            )
        )
    ),
    jetSrc = cms.InputTag("hltAK4PFJets"),
    maxJetAbsEta = cms.double(99.0),
    minJetPt = cms.double(-1.0),
    outputSelection = cms.string('pt > 0.5'),
    ranking = cms.VPSet(
        cms.PSet(
            name = cms.string('ChargedPFCandidate'),
            plugin = cms.string('PFRecoTauChargedHadronStringQuality'),
            selection = cms.string("algoIs(\'kChargedPFCandidate\')"),
            selectionFailValue = cms.double(1000.0),
            selectionPassFunction = cms.string('-pt')
        ),
        cms.PSet(
            name = cms.string('ChargedPFCandidate'),
            plugin = cms.string('PFRecoTauChargedHadronStringQuality'),
            selection = cms.string("algoIs(\'kPFNeutralHadron\')"),
            selectionFailValue = cms.double(1000.0),
            selectionPassFunction = cms.string('-pt')
        )
    ),
    verbosity = cms.int32(0)
)

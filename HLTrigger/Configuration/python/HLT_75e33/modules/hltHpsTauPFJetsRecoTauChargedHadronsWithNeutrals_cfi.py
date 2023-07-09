import FWCore.ParameterSet.Config as cms

hltHpsTauPFJetsRecoTauChargedHadronsWithNeutrals = cms.EDProducer( "PFRecoTauChargedHadronProducer",
    ranking = cms.VPSet( 
      cms.PSet(  selectionFailValue = cms.double( 1000.0 ),
        plugin = cms.string( "PFRecoTauChargedHadronStringQuality" ),
        selection = cms.string( "algoIs(\'kChargedPFCandidate\')" ),
        name = cms.string( "ChargedPFCandidate" ),
        selectionPassFunction = cms.string( "-pt" )
      ),
      cms.PSet(  selectionFailValue = cms.double( 1000.0 ),
        plugin = cms.string( "PFRecoTauChargedHadronStringQuality" ),
        selection = cms.string( "algoIs(\'kPFNeutralHadron\')" ),
        name = cms.string( "ChargedPFCandidate" ),
        selectionPassFunction = cms.string( "-pt" )
      )
    ),
    verbosity = cms.int32( 0 ),
    maxJetAbsEta = cms.double( 99.0 ),
    outputSelection = cms.string( "pt > 0.5" ),
    minJetPt = cms.double( -1.0 ),
    jetSrc = cms.InputTag( "hltAK4PFJets" ),
    builders = cms.VPSet( 
      cms.PSet(  minBlockElementMatchesNeutralHadron = cms.int32( 2 ),
        dRmergeNeutralHadronWrtNeutralHadron = cms.double( 0.01 ),
        dRmergePhotonWrtNeutralHadron = cms.double( 0.01 ),
        dRmergePhotonWrtOther = cms.double( 0.005 ),
        qualityCuts = cms.PSet( 
          vertexTrackFiltering = cms.bool( False ),
          primaryVertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),
          recoverLeadingTrk = cms.bool( False ),
          signalQualityCuts = cms.PSet( 
            minNeutralHadronEt = cms.double( 30.0 ),
            maxDeltaZ = cms.double( 0.2 ),
            minTrackPt = cms.double( 0.0 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 1000.0 ),
            maxTransverseImpactParameter = cms.double( 0.2 ),
            useTracksInsteadOfPFHadrons = cms.bool( False )
          ),
          vxAssocQualityCuts = cms.PSet( 
            minTrackPt = cms.double( 0.0 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 1000.0 ),
            maxTransverseImpactParameter = cms.double( 0.2 ),
            useTracksInsteadOfPFHadrons = cms.bool( False )
          ),
          pvFindingAlgo = cms.string( "closestInDeltaZ" )
        ),
        dRmergeNeutralHadronWrtOther = cms.double( 0.005 ),
        maxUnmatchedBlockElementsNeutralHadron = cms.int32( 1 ),
        dRmergePhotonWrtElectron = cms.double( 0.005 ),
        minMergeGammaEt = cms.double( 0.0 ),
        minBlockElementMatchesPhoton = cms.int32( 2 ),
        dRmergePhotonWrtChargedHadron = cms.double( 0.005 ),
        plugin = cms.string( "PFRecoTauChargedHadronFromPFCandidatePlugin" ),
        dRmergeNeutralHadronWrtChargedHadron = cms.double( 0.005 ),
        minMergeChargedHadronPt = cms.double( 100.0 ),
        maxUnmatchedBlockElementsPhoton = cms.int32( 1 ),
        name = cms.string( "chargedPFCandidates" ),
        dRmergeNeutralHadronWrtElectron = cms.double( 0.05 ),
        chargedHadronCandidatesParticleIds = cms.vint32( 1, 2, 3 ),
        minMergeNeutralHadronEt = cms.double( 0.0 )
      ),
      cms.PSet(  minBlockElementMatchesNeutralHadron = cms.int32( 2 ),
        dRmergeNeutralHadronWrtNeutralHadron = cms.double( 0.01 ),
        dRmergePhotonWrtNeutralHadron = cms.double( 0.01 ),
        dRmergePhotonWrtOther = cms.double( 0.005 ),
        qualityCuts = cms.PSet( 
          vertexTrackFiltering = cms.bool( False ),
	  primaryVertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),
          recoverLeadingTrk = cms.bool( False ),
          signalQualityCuts = cms.PSet( 
            minNeutralHadronEt = cms.double( 30.0 ),
            maxDeltaZ = cms.double( 0.2 ),
            minTrackPt = cms.double( 0.0 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 1000.0 ),
            maxTransverseImpactParameter = cms.double( 0.2 ),
            useTracksInsteadOfPFHadrons = cms.bool( False )
          ),
          vxAssocQualityCuts = cms.PSet( 
            minTrackPt = cms.double( 0.0 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 1000.0 ),
            maxTransverseImpactParameter = cms.double( 0.2 ),
            useTracksInsteadOfPFHadrons = cms.bool( False )
          ),
          pvFindingAlgo = cms.string( "closestInDeltaZ" )
        ),
        dRmergeNeutralHadronWrtOther = cms.double( 0.005 ),
        dRmergePhotonWrtElectron = cms.double( 0.005 ),
        minMergeGammaEt = cms.double( 0.0 ),
        dRmergePhotonWrtChargedHadron = cms.double( 0.005 ),
        plugin = cms.string( "PFRecoTauChargedHadronFromPFCandidatePlugin" ),
        dRmergeNeutralHadronWrtChargedHadron = cms.double( 0.005 ),
        minMergeChargedHadronPt = cms.double( 0.0 ),
        maxUnmatchedBlockElementsPhoton = cms.int32( 1 ),
        name = cms.string( "PFNeutralHadrons" ),
        dRmergeNeutralHadronWrtElectron = cms.double( 0.05 ),
        chargedHadronCandidatesParticleIds = cms.vint32( 5 ),
        minMergeNeutralHadronEt = cms.double( 0.0 ),
        maxUnmatchedBlockElementsNeutralHadron = cms.int32( 1 ),
        minBlockElementMatchesPhoton = cms.int32( 2 )
      )
    )
)

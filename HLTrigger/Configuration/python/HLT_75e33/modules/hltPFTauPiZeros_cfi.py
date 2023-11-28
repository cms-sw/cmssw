import FWCore.ParameterSet.Config as cms

hltPFTauPiZeros = cms.EDProducer( "RecoTauPiZeroProducer",
    massHypothesis = cms.double( 0.136 ),
    ranking = cms.VPSet( 
      cms.PSet(  selectionFailValue = cms.double( 1000.0 ),
        plugin = cms.string( "RecoTauPiZeroStringQuality" ),
        selection = cms.string( "algoIs('kStrips')" ),
        name = cms.string( "InStrip" ),
        selectionPassFunction = cms.string( "abs(mass() - 0.13579)" )
      )
    ),
    verbosity = cms.int32( 0 ),
    maxJetAbsEta = cms.double( 99.0 ),
    outputSelection = cms.string( "pt > 0" ),
    minJetPt = cms.double( -1.0 ),
    jetSrc = cms.InputTag( "hltAK4PFJets" ),
    builders = cms.VPSet( 
      cms.PSet(  minGammaEtStripSeed = cms.double( 0.5 ),
        applyElecTrackQcuts = cms.bool( False ),
        stripCandidatesParticleIds = cms.vint32( 2, 4 ),
        makeCombinatoricStrips = cms.bool( False ),
        stripPhiAssociationDistance = cms.double( 0.2 ),
        qualityCuts = cms.PSet( 
          vertexTrackFiltering = cms.bool( False ),
          primaryVertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),
          recoverLeadingTrk = cms.bool( False ),
          signalQualityCuts = cms.PSet( 
            maxDeltaZ = cms.double( 0.2 ),
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
        maxStripBuildIterations = cms.int32( -1 ),
        updateStripAfterEachDaughter = cms.bool( False ),
        stripEtaAssociationDistance = cms.double( 0.05 ),
        minStripEt = cms.double( 1.0 ),
        plugin = cms.string( "RecoTauPiZeroStripPlugin2" ),
        minGammaEtStripAdd = cms.double( 0.0 ),
        name = cms.string( "s" )
      )
    )
)

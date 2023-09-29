import FWCore.ParameterSet.Config as cms

hltHpsPFTauMediumAbsoluteChargedIsolationDiscriminator = cms.EDProducer( "PFRecoTauDiscriminationByIsolation",
    storeRawFootprintCorrection = cms.bool( False ),
    PFTauProducer = cms.InputTag( "hltHpsPFTauProducer" ),    
    storeRawOccupancy = cms.bool( False ),
    maximumSumPtCut = cms.double( 3.7 ),
    qualityCuts = cms.PSet( 
      vertexTrackFiltering = cms.bool( False ),
      isolationQualityCuts = cms.PSet( 
        maxDeltaZ = cms.double( 0.2 ),
        minTrackPt = cms.double( 0.5 ),
        minGammaEt = cms.double( 0.5 ),
        minTrackHits = cms.uint32( 3 ),
        minTrackPixelHits = cms.uint32( 0 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.1 ),
        useTracksInsteadOfPFHadrons = cms.bool( False )
      ),
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
        useTracksInsteadOfPFHadrons = cms.bool( False ),
        minNeutralHadronEt = cms.double( 1.0 )
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
    minTauPtForNoIso = cms.double( -99.0 ),
    maxAbsPhotonSumPt_outsideSignalCone = cms.double( 1.0E9 ),
    vertexSrc = cms.InputTag( "NotUsed" ),
    applySumPtCut = cms.bool( True ),
    rhoConeSize = cms.double( 0.357 ),
    ApplyDiscriminationByTrackerIsolation = cms.bool( True ),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool( False ),
    rhoProducer = cms.InputTag( "NotUsed" ),
    enableHGCalWorkaround = cms.bool( False ),
    footprintCorrections = cms.VPSet( 
      cms.PSet(  offset = cms.string( "0.0" ),
        selection = cms.string( "decayMode() = 0" )
      ),
      cms.PSet(  offset = cms.string( "0.0" ),
        selection = cms.string( "decayMode() = 1 || decayMode() = 2" )
      ),
      cms.PSet(  offset = cms.string( "2.7" ),
        selection = cms.string( "decayMode() = 5" )
      ),
      cms.PSet(  offset = cms.string( "0.0" ),
        selection = cms.string( "decayMode() = 6" )
      ),
      cms.PSet(  offset = cms.string( "max(2.0, 0.22*pt() - 2.0)" ),
        selection = cms.string( "decayMode() = 10" )
      )
    ),
    deltaBetaFactor = cms.string( "0.38" ),
    applyFootprintCorrection = cms.bool( False ),
    UseAllPFCandsForWeights = cms.bool( False ),
    relativeSumPtCut = cms.double( 0.03 ),
    Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
    maximumOccupancy = cms.uint32( 0 ),
    verbosity = cms.int32( 0 ),
    applyOccupancyCut = cms.bool( False ),
    applyDeltaBetaCorrection = cms.bool( False ),
    applyRelativeSumPtCut = cms.bool( False ),
    storeRawPUsumPt = cms.bool( False ),
    applyPhotonPtSumOutsideSignalConeCut = cms.bool( False ),
    deltaBetaPUTrackPtCutOverride = cms.bool( True ),
    ApplyDiscriminationByWeightedECALIsolation = cms.bool( False ),
    storeRawSumPt = cms.bool( False ),
    ApplyDiscriminationByECALIsolation = cms.bool( False ),
    applyRhoCorrection = cms.bool( False ),
    WeightECALIsolation = cms.double( 0.33333 ),
    rhoUEOffsetCorrection = cms.double( 0.0 ),
    maxRelPhotonSumPt_outsideSignalCone = cms.double( 0.1 ),
    deltaBetaPUTrackPtCutOverride_val = cms.double( 0.5 ),
    isoConeSizeForDeltaBeta = cms.double( 0.3 ),
    relativeSumPtOffset = cms.double( 0.0 ),
    customOuterCone = cms.double( -1.0 ),
    particleFlowSrc = cms.InputTag( "particleFlowTmp" )  
)

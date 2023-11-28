import FWCore.ParameterSet.Config as cms

hltHpsPFTauBasicDiscriminatorsdR03ForDeepTau = cms.EDProducer( "PFRecoTauDiscriminationByIsolationContainer",
    PFTauProducer = cms.InputTag( "hltHpsPFTauProducer" ),
    qualityCuts = cms.PSet( 
      signalQualityCuts = cms.PSet( 
        minTrackPt = cms.double( 0.5 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.1 ),
        maxDeltaZ = cms.double( 0.4 ),
        maxDeltaZToLeadTrack = cms.double( -1.0 ),
        minTrackVertexWeight = cms.double( -1.0 ),
        minTrackPixelHits = cms.uint32( 0 ),
        minTrackHits = cms.uint32( 3 ),
        minGammaEt = cms.double( 1.0 ),
        minNeutralHadronEt = cms.double( 30.0 ),
        useTracksInsteadOfPFHadrons = cms.bool( False )
      ),
      isolationQualityCuts = cms.PSet( 
        minTrackPt = cms.double( 1.0 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.03 ),
        maxDeltaZ = cms.double( 0.2 ),
        maxDeltaZToLeadTrack = cms.double( -1.0 ),
        minTrackVertexWeight = cms.double( -1.0 ),
        minTrackPixelHits = cms.uint32( 0 ),
        minTrackHits = cms.uint32( 8 ),
        minGammaEt = cms.double( 1.5 ),
        useTracksInsteadOfPFHadrons = cms.bool( False )
      ),
      vxAssocQualityCuts = cms.PSet( 
        minTrackPt = cms.double( 0.5 ),
        maxTrackChi2 = cms.double( 100.0 ),
        maxTransverseImpactParameter = cms.double( 0.1 ),
        minTrackVertexWeight = cms.double( -1.0 ),
        minTrackPixelHits = cms.uint32( 0 ),
        minTrackHits = cms.uint32( 3 ),
        minGammaEt = cms.double( 1.0 )
      ),
      primaryVertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),
      pvFindingAlgo = cms.string( "closestInDeltaZ" ),
      vertexTrackFiltering = cms.bool( False ),
      recoverLeadingTrk = cms.bool( False ),
      leadingTrkOrPFCandOption = cms.string( "leadPFCand" )
    ),
    minTauPtForNoIso = cms.double( -99.0 ),
    vertexSrc = cms.InputTag( "hltPhase2PixelVertices" ),
    rhoConeSize = cms.double( 0.5 ),
    rhoProducer = cms.InputTag( "hltFixedGridRhoProducerFastjetAllTau" ),
    footprintCorrections = cms.VPSet( 
      cms.PSet(  selection = cms.string( "decayMode() = 0" ),
        offset = cms.string( "0.0" )
      ),
      cms.PSet(  selection = cms.string( "decayMode() = 1 || decayMode() = 2" ),
        offset = cms.string( "0.0" )
      ),
      cms.PSet(  selection = cms.string( "decayMode() = 5" ),
        offset = cms.string( "2.7" )
      ),
      cms.PSet(  selection = cms.string( "decayMode() = 6" ),
        offset = cms.string( "0.0" )
      ),
      cms.PSet(  selection = cms.string( "decayMode() = 10" ),
        offset = cms.string( "max(2.0, 0.22*pt() - 2.0)" )
      )
    ),
    deltaBetaFactor = cms.string( "0.2000" ),
    applyFootprintCorrection = cms.bool( False ),
    Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
    verbosity = cms.int32( 0 ),
    deltaBetaPUTrackPtCutOverride = cms.bool( True ),
    applyRhoCorrection = cms.bool( False ),
    WeightECALIsolation = cms.double( 1.0 ),
    rhoUEOffsetCorrection = cms.double( 1.0 ),
    deltaBetaPUTrackPtCutOverride_val = cms.double( 0.5 ),
    isoConeSizeForDeltaBeta = cms.double( 0.8 ),
    customOuterCone = cms.double( 0.3 ),
    particleFlowSrc = cms.InputTag( "particleFlowTmp" ),
    IDdefinitions = cms.VPSet( 
      cms.PSet(  IDname = cms.string( "ChargedIsoPtSum" ),
        ApplyDiscriminationByTrackerIsolation = cms.bool( True ),
        storeRawSumPt = cms.bool( True )
      ),
      cms.PSet(  IDname = cms.string( "NeutralIsoPtSum" ),
        ApplyDiscriminationByECALIsolation = cms.bool( True ),
        storeRawSumPt = cms.bool( True )
      ),
      cms.PSet(  IDname = cms.string( "NeutralIsoPtSumWeight" ),
        ApplyDiscriminationByWeightedECALIsolation = cms.bool( True ),
        storeRawSumPt = cms.bool( True ),
        UseAllPFCandsForWeights = cms.bool( True )
      ),
      cms.PSet(  IDname = cms.string( "TauFootprintCorrection" ),
        storeRawFootprintCorrection = cms.bool( True )
      ),
      cms.PSet(  IDname = cms.string( "PhotonPtSumOutsideSignalCone" ),
        storeRawPhotonSumPt_outsideSignalCone = cms.bool( True )
      ),
      cms.PSet(  IDname = cms.string( "PUcorrPtSum" ),
        applyDeltaBetaCorrection = cms.bool( True ),
        storeRawPUsumPt = cms.bool( True )
      )
    ),
    IDWPdefinitions = cms.VPSet( 
    )
)

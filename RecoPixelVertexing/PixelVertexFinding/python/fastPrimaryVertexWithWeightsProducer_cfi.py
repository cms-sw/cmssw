import FWCore.ParameterSet.Config as cms

fastPrimaryVertices = cms.EDProducer( "FastPrimaryVertexWithWeightsProducer",
    maxZ = cms.double( 19.0 ),
    clusters = cms.InputTag( "hltSiPixelClusters" ),
    pixelCPE = cms.string( "hltESPPixelCPEGeneric" ),
    beamSpot = cms.InputTag( "hltOnlineBeamSpot" ),
    jets = cms.InputTag( "hltCaloJetL1FastJetCorrected" ),
  
    njets     = cms.int32( 999 )	,
    maxJetEta = cms.double( 2.6 ), 
    minJetPt  = cms.double( 40. ),

    barrel = cms.bool(True),	
    maxSizeX = cms.double( 2.1 ),
    maxDeltaPhi = cms.double( 0.21 ),
    PixelCellHeightOverWidth  = cms.double( 1.8 ),
    weight_charge_down   = cms.double(11*1000),
    weight_charge_up   = cms.double(190*1000),
    maxSizeY_q  = cms.double( 2 ),	
    minSizeY_q  = cms.double( -0.6 ),	
 
    weight_dPhi     = cms.double( 0.13888888 ),	
    weight_SizeX1    = cms.double(0.88),
    weight_rho_up    = cms.double( 22),
    weight_charge_peak   = cms.double(22*1000),
    peakSizeY_q  = cms.double( 1.0 ),	

    endCap = cms.bool(True),	
    minJetEta_EC  = cms.double( 1.3 ),	
    maxJetEta_EC  = cms.double( 2.6 ),	
    maxDeltaPhi_EC  = cms.double( 0.14 ),
    EC_weight     = cms.double( 0.008 ),
    weight_dPhi_EC     = cms.double( 0.064516129 ),	
    
    zClusterWidth_step1    = cms.double( 2.0),

    zClusterWidth_step2    = cms.double( 0.65), 
    zClusterSearchArea_step2    = cms.double( 3.0),
    weightCut_step2    = cms.double( 0.05),

    zClusterWidth_step3    = cms.double( 0.3), 
    zClusterSearchArea_step3    = cms.double( 0.55),
    weightCut_step3    = cms.double( 0.1),

)



import FWCore.ParameterSet.Config as cms

histoHelper4hltTracking = cms.PSet(

    # to be added here all the other histogram settings
    #
    minEta     = cms.double(-2.5),
    maxEta     = cms.double( 2.5),
    nintEta    = cms.int32(50),
    useFabsEta = cms.bool(False),

    minPt    = cms.double(   0.1),
    maxPt    = cms.double(100.0),
    nintPt   = cms.int32(1000),
    useInvPt = cms.bool(False),
    useLogPt =cms.untracked.bool(False),

#    minHit  = cms.double(-0.5),                            
#    maxHit  = cms.double(40.5),
#    nintHit = cms.int32(41),
#
#    minLayers  = cms.double(-0.5),                            
#    maxLayers  = cms.double(15.5),
#    nintLayers = cms.int32(16),

    minPhi  = cms.double(-3.1416),
    maxPhi  = cms.double( 3.1416),
    nintPhi = cms.int32(36),

    minDxy  = cms.double(-15.0),
    maxDxy  = cms.double( 15.0),
    nintDxy = cms.int32(600),

    minDz  = cms.double(-30.0),
    maxDz  = cms.double( 30.0),
    nintDz = cms.int32(60),

#    # TP originating vertical position
#    minVertpos  = cms.double( 0.0),
#    maxVertpos  = cms.double(60.0),
#    nintVertpos = cms.int32(60),
#    #
#    # TP originating z position
#    minZpos  = cms.double(-30.0),
#    maxZpos  = cms.double( 30.0),
#    nintZpos = cms.int32(60),                               
#
#    #
#    #parameters for resolution plots
#    ptRes_rangeMin = cms.double(-0.1),
#    ptRes_rangeMax = cms.double( 0.1),
#    ptRes_nbin = cms.int32(100),                                   
#
#    phiRes_rangeMin = cms.double(-0.01),
#    phiRes_rangeMax = cms.double( 0.01),
#    phiRes_nbin = cms.int32(300),                                   
#
#    cotThetaRes_rangeMin = cms.double(-0.02),
#    cotThetaRes_rangeMax = cms.double( 0.02),
#    cotThetaRes_nbin = cms.int32(300),                                   
#
#    dxyRes_rangeMin = cms.double(-0.1),
#    dxyRes_rangeMax = cms.double( 0.1),
#    dxyRes_nbin = cms.int32(500),                                   
#
#    dzRes_rangeMin = cms.double(-0.05),
#    dzRes_rangeMax = cms.double( 0.05),
#    dzRes_nbin = cms.int32(150),                                   


)    

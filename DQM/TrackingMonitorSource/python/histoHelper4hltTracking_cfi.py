import FWCore.ParameterSet.Config as cms

histoHelper4hltTracking = cms.PSet(

    minEta     = cms.double(-2.5),
    maxEta     = cms.double( 2.5),
    nintEta    = cms.int32(50),
    useFabsEta = cms.bool(False),

    minPt    = cms.double(   0.1),
    maxPt    = cms.double(100.0),
    nintPt   = cms.int32(1000),
    useInvPt = cms.bool(False),
    useLogPt =cms.untracked.bool(False),

    minPhi  = cms.double(-3.1416),
    maxPhi  = cms.double( 3.1416),
    nintPhi = cms.int32(36),

    minDxy  = cms.double(-15.0),
    maxDxy  = cms.double( 15.0),
    nintDxy = cms.int32(600),

    minDz  = cms.double(-30.0),
    maxDz  = cms.double( 30.0),
    nintDz = cms.int32(60),

    #
    #  parameters for resolution plots
    #
    ptRes_rangeMin = cms.double(-0.1),
    ptRes_rangeMax = cms.double( 0.1),
    ptRes_nbin = cms.int32(100),                                   

    phiRes_rangeMin = cms.double(-0.01),
    phiRes_rangeMax = cms.double( 0.01),
    phiRes_nbin = cms.int32(300),                                   

    etaRes_rangeMin = cms.double(-0.01),
    etaRes_rangeMax = cms.double( 0.01),
    etaRes_nbin = cms.int32(300),                                   

    dxyRes_rangeMin = cms.double(-0.05),
    dxyRes_rangeMax = cms.double( 0.05),
    dxyRes_nbin = cms.int32(500),                                   

    dzRes_rangeMin = cms.double(-0.05),
    dzRes_rangeMax = cms.double( 0.05),
    dzRes_nbin = cms.int32(150),                                   


)    

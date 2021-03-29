import FWCore.ParameterSet.Config as cms

bmtfKalmanTrackingSettings = cms.PSet(
    verbose = cms.bool(False),  # 
    lutFile = cms.string("L1Trigger/L1TMuon/data/bmtf_luts/kalmanLUTs.root"),
    initialK = cms.vdouble(-1.196,-1.581,-2.133,-2.263),
    initialK2 = cms.vdouble(-3.26e-4,-7.165e-4,2.305e-3,-5.63e-3),
#    eLoss = cms.vdouble(-2.85e-4,-6.21e-5,-1.26e-4,-1.23e-4), 
    eLoss = cms.vdouble(+0.000765,0,0,0), 

    aPhi = cms.vdouble(1.942, .01511, .01476, .009799),
    aPhiB = cms.vdouble(-1.508,-0.1237,-0.1496,-0.1333),
    aPhiBNLO = cms.vdouble(0.000331,0,0,0),

    bPhi = cms.vdouble(-1,.18245,.20898,.17286),
    bPhiB = cms.vdouble(-1,1.18245,1.20898,1.17286),
    phiAt2 = cms.double(0.15918),
    etaLUT0 = cms.vdouble(8.946,7.508,6.279,6.399),
    etaLUT1 = cms.vdouble(0.159,0.116,0.088,0.128),
    #generic cuts
    chiSquare = cms.vdouble(0.0,0.109375,0.234375,0.359375),   
    chiSquareCutPattern = cms.vint32(7,11,13,14,15),
    chiSquareCutCurvMax = cms.vint32(2500,2500,2500,2500,2500),
    chiSquareCut = cms.vint32(126,126,126,126,126),


    #vertex cuts
    trackComp = cms.vdouble(1.75,1.25,0.625,0.250),   
    trackCompErr1 = cms.vdouble(2.0,2.0,2.0,2.0),   
    trackCompErr2 = cms.vdouble(0.218750,0.218750,0.218750,0.3125),   
    trackCompCutPattern = cms.vint32(3,5,6,9,10,12),   
    trackCompCutCurvMax = cms.vint32(34,34,34,34,34,34),   #this is shifted<<4
    trackCompCut        = cms.vint32(15,15,15,15,15,15),  
    chiSquareCutTight   = cms.vint32(40,126,60,126,126,126),  

    combos4=cms.vint32(9,10,11,12,13,14,15),
    combos3=cms.vint32(5,6,7),
    combos2=cms.vint32(3),
    combos1=cms.vint32(), #for future possible usage

    useOfflineAlgo = cms.bool(False),   
    ###Only for the offline algo -not in firmware --------------------
    mScatteringPhi = cms.vdouble(2.49e-3,5.47e-5,3.49e-5,1.37e-5),
    mScatteringPhiB = cms.vdouble(7.22e-3,3.461e-3,4.447e-3,4.12e-3),
    pointResolutionPhi = cms.double(1.),
    pointResolutionPhiB = cms.double(500.),
    pointResolutionPhiBH = cms.vdouble(151., 173., 155., 153.),
    pointResolutionPhiBL = cms.vdouble(17866., 19306., 23984., 23746.),
    pointResolutionVertex = cms.double(1.)
)



simKBmtfDigis = cms.EDProducer("L1TMuonBarrelKalmanTrackProducer",
    src = cms.InputTag("simKBmtfStubs"),
    bx = cms.vint32(-2,-1,0,1,2),
#    bx = cms.vint32(0),
    algoSettings = bmtfKalmanTrackingSettings,
    trackFinderSettings = cms.PSet(
        sectorsToProcess = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11),
        verbose = cms.int32(0),
        sectorSettings = cms.PSet(
#            verbose = cms.int32(1),
            verbose = cms.int32(0),
            wheelsToProcess = cms.vint32(-2,-1,0,1,2),
            regionSettings = cms.PSet(
                verbose=cms.int32(0)
            )
        )
        
    )
)

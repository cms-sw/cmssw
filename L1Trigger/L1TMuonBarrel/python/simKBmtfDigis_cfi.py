import FWCore.ParameterSet.Config as cms

bmtfKalmanTrackingSettings = cms.PSet(
    verbose = cms.bool(False),
    lutFile = cms.string("L1Trigger/L1TMuonBarrel/data/kalmanLUTs.root"),
    initialK = cms.vdouble(-1.196,-1.581,-2.133,-2.263),
    initialK2 = cms.vdouble(-3.26e-4,-7.165e-4,2.305e-3,-5.63e-3),
#    eLoss = cms.vdouble(-2.85e-4,-6.21e-5,-1.26e-4,-1.23e-4), 
    eLoss = cms.vdouble(+3.825e-4,0,0,0), 
    aPhi = cms.vdouble(1.942,0.032,0.028,0.016),
#    aPhiB = cms.vdouble(-0.740,-0.123,-0.174,-0.144),
#    aPhiBNLO = cms.vdouble(1.758e-4,0,0,0),
    aPhiB = cms.vdouble(-0.754,-0.123,-0.174,-0.144),
    aPhiBNLO = cms.vdouble(1.657e-4,0,0,0),

    bPhi = cms.vdouble(-1,0.154,0.173,0.147),
    bPhiB = cms.vdouble(-1,1.154,1.173,1.147),
    etaLUT0 = cms.vdouble(8.946,7.508,6.279,6.399),
    etaLUT1 = cms.vdouble(0.159,0.116,0.088,0.128),
    chiSquare = cms.vdouble(0.0,0.095,0.232,0.356),   
#    chiSquareCutPattern = cms.vint32(3,6),
#    chiSquareCutCurvMax = cms.vint32(273,273),
    globalChi2Cut = cms.uint32(510),
    chiSquareCutPattern = cms.vint32(3,6,12),
    chiSquareCutCurvMax = cms.vint32(273,273,273),
    chiSquareCut = cms.vint32(100,128,128),
#    chiSquareCut = cms.vint32(1000,1000,1000,1000,1000,1000,1200,1200,1200,1200,1200,8192,8192,8192),


    combos4=cms.vint32(9,10,11,12,13,14,15),
    combos3=cms.vint32(5,6,7),
    combos2=cms.vint32(3),
    combos1=cms.vint32(), #for future possible usage


    useOfflineAlgo = cms.bool(False),
    
    ###Only for the offline algo -not in firmware --------------------

#    mScatteringPhi = cms.vdouble(0.9322,8.384e-5,5.406e-5,1.786e-5),
#    mScatteringPhiB = cms.vdouble(7.92e-3,3.580e-3,6.634e-3,5.478e-3),
#    pointResolutionPhi = cms.double(1.),
#    pointResolutionPhiB = cms.double(500.),

    mScatteringPhi = cms.vdouble(2.49e-3,5.47e-5,3.49e-5,1.37e-5),
    mScatteringPhiB = cms.vdouble(7.22e-3,3.461e-3,4.447e-3,4.12e-3),

#    mScatteringPhi = cms.vdouble(0,0,0,0),
#    mScatteringPhiB = cms.vdouble(0,0,0,0),

    pointResolutionPhi = cms.double(1.),
    pointResolutionPhiB = cms.double(500.),
    pointResolutionVertex = cms.double(1.)
)



simKBmtfDigis = cms.EDProducer("L1TMuonBarrelKalmanTrackProducer",
    src = cms.InputTag("simKBmtfStubs"),
    bx = cms.vint32(0),
    algoSettings = bmtfKalmanTrackingSettings,
    trackFinderSettings = cms.PSet(
        sectorsToProcess = cms.vint32(0,1,2,3,4,5,6,7,8,9,10,11),
        verbose = cms.int32(0),
        sectorSettings = cms.PSet(
            verbose = cms.int32(0),
            wheelsToProcess = cms.vint32(-2,-1,0,1,2),
            regionSettings = cms.PSet(
                verbose=cms.int32(0)
            )
        )
        
    )
)

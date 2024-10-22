import FWCore.ParameterSet.Config as cms


gmtKMTFMuons = cms.EDProducer('Phase2L1TGMTKMTFProducer',
                                 stubs      = cms.InputTag('gmtStubs','kmtf'),
                                 verbose    = cms.int32(0),
                                 algo       = cms.PSet(
                                     verbose = cms.bool(False),
                                     lutFile = cms.string("L1Trigger/Phase2L1GMT/data/packedGainLUTs.root"),
                                     initialK = cms.vdouble(-0.4576229536749278, -0.6364802777566145, -1.0305030909883524, -1.7272067322624118),
                                     initialK2 = cms.vdouble(-6.442002637356136e-05, -9.582709649965545e-05, -0.0002741064246218815, -0.0014910074450869175),
                                     eLoss = cms.vdouble(6.77765e-05,0,0,0),
                                     
                                     aPhi = cms.vdouble(5.6533349391874275, 0.03477876333443834, 0.032506522838098864, 0.024752809174909053),
                                     aPhiB = cms.vdouble(-2.02, -0.2994087741381382, -0.4033668521165302, -0.3592231728688621),
                                     aPhiBNLO = cms.vdouble(9.04133e-05,0,0,0),

                                     bPhi = cms.vdouble(-1,.18245,.20898,.17286),
                                     bPhiB = cms.vdouble(-1,1.18245,1.20898,1.17286),
                                     phiAt2 = cms.double(0.17917991867483288),
                                     etaLUT0 = cms.vdouble(8.946,7.508,6.279,6.399),
                                     etaLUT1 = cms.vdouble(0.159,0.116,0.088,0.128),
                                     
                                     #Chi2 Calculation coefficients
                                     
                                     chiSquareDisp1 = cms.vdouble(0.241,0.575,0.892),
                                     chiSquareDisp2 = cms.vdouble(0.347,0.707),
                                     chiSquareDisp3 = cms.vdouble(0.336),

                                     chiSquareErrADisp1 = cms.vint32(21,23,45),
                                     chiSquareErrADisp2 = cms.vint32(14,28),
                                     chiSquareErrADisp3 = cms.vint32(29),
                                     chiSquareErrBDisp1 = cms.vdouble(5*0.108,5*0.161,5*0.148),
                                     chiSquareErrBDisp2 = cms.vdouble(5*0.146,5*0.189),
                                     chiSquareErrBDisp3 = cms.vdouble(5*0.100),

                                     chiSquarePrompt1 = cms.vdouble(-0.323,-0.508,-0.578),
                                     chiSquarePrompt2 = cms.vdouble(-0.228,-0.343),
                                     chiSquarePrompt3 = cms.vdouble(-0.100),
                                     
                                     chiSquareErrAPrompt1 = cms.vint32(21,24,33),
                                     chiSquareErrAPrompt2 = cms.vint32(18,14),
                                     chiSquareErrAPrompt3 = cms.vint32(21),
                                     chiSquareErrBPrompt1 = cms.vdouble(5*0.009,5*0.016,5*0.012),
                                     chiSquareErrBPrompt2 = cms.vdouble(5*0.013,5*0.055),
                                     chiSquareErrBPrompt3 = cms.vdouble(5*0.01),
                                     chiSquareCutDispPattern = cms.vint32(),
                                     chiSquareCutOffDisp = cms.vint32(),
                                     chiSquareCutDisp = cms.vint32(),
                                     chiSquareCutPromptPattern = cms.vint32(),
                                     chiSquareCutOffPrompt = cms.vint32(),
                                     chiSquareCutPrompt = cms.vint32(),
                                     combos4=cms.vint32(9,10,11,12,13,14,15),
                                     combos3=cms.vint32(5,6,7),
                                     combos2=cms.vint32(3),
                                     combos1=cms.vint32(), #for future possible usage
                                     useOfflineAlgo = cms.bool(False),   
                                     ###Only for the offline algo -not in firmware --------------------
                                     mScatteringPhi = cms.vdouble(0.1169021113298598, 0.00016777763395543814, 0.0004322078772344548, 0.00024985881710722107),
                                     mScatteringPhiB = cms.vdouble(.0522762, 0.01762000062188365, 0.03508319015441297, 0.03126825551530328),
                                     pointResolutionPhi = cms.double(1.),
                                     pointResolutionPhiB = cms.double(12493.7429036),
                                     pointResolutionPhiBH = cms.vdouble(19925.62953113343, 15583.06791339368, 10258.11768352221, 15462.112839170433),
                                     pointResolutionPhiBL = cms.vdouble(161519.85395846734, 155051.58394241595, 149693.88179343447, 174896.46766622085),
                                     pointResolutionVertex = cms.double(1.),
                                     curvResolution1 = cms.vdouble(1, 2.36097e+03, 8.73003e+02, 2.58138e5),
                                     curvResolution2 = cms.vdouble(1, 4.903692e+00, 4.87941e+01, 0)),
                                 Nprompt    = cms.uint32(12),
                                 Ndisplaced = cms.uint32(12)
)




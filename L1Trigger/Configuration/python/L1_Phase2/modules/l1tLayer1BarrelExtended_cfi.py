import FWCore.ParameterSet.Config as cms

l1tLayer1BarrelExtended = cms.EDProducer("L1TCorrelatorLayer1Producer",
    boards = cms.VPSet(
        cms.PSet(
            regions = cms.vuint32(
                5, 14, 23, 32, 41,
                50, 6, 15, 24, 33,
                42, 51, 7, 16, 25,
                34, 43, 52
            )
        ),
        cms.PSet(
            regions = cms.vuint32(
                8, 17, 26, 35, 44,
                53, 0, 9, 18, 27,
                36, 45, 1, 10, 19,
                28, 37, 46
            )
        ),
        cms.PSet(
            regions = cms.vuint32(
                2, 11, 20, 29, 38,
                47, 3, 12, 21, 30,
                39, 48, 4, 13, 22,
                31, 40, 49
            )
        )
    ),
    caloSectors = cms.VPSet(cms.PSet(
        etaBoundaries = cms.vdouble(-1.5, 0, 1.5),
        phiSlices = cms.uint32(6),
        phiZero = cms.double(0.17453292519943295)
    )),
    emPtCut = cms.double(0.5),
    gctClusters = cms.InputTag("l1tPhase2CaloToCorrelatorTM18","DigitizedCaloToCorrelatorTM18"),
    gctEmInputConversionAlgo = cms.string('Emulator'),
    gctEmInputConversionParameters = cms.PSet(
        gctEmCorrector = cms.string('L1Trigger/Phase2L1ParticleFlow/data/emcorr_barrel.root'),
        gctEmResol = cms.PSet(
            etaBins = cms.vdouble(0.7, 1.2, 1.6),
            kind = cms.string('calo'),
            offset = cms.vdouble(0.873, 1.081, 1.563),
            scale = cms.vdouble(0.011, 0.015, 0.012)
        )
    ),
    gctHadInputConversionAlgo = cms.string('Emulator'),
    gctHadInputConversionParameters = cms.PSet(
        gctHadCorrector = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel.root'),
        gctHadResol = cms.PSet(
            etaBins = cms.vdouble(0.7, 1.2, 1.6),
            kind = cms.string('calo'),
            offset = cms.vdouble(3.675, 2.694, 1.64),
            scale = cms.vdouble(0.121, 0.156, 0.292)
        )
    ),
    hadPtCut = cms.double(1.0),
    muonInputConversionAlgo = cms.string('Emulator'),
    muons = cms.InputTag("l1tSAMuonsGmt","prompt"),
    nVtx = cms.int32(1),
    pfAlgo = cms.string('PFAlgo3'),
    pfAlgoParameters = cms.PSet(
        caloResolution = cms.PSet(
            etaBins = cms.vdouble(0.7, 1.2, 1.6),
            offset = cms.vdouble(2.909, 2.864, 0.294),
            scale = cms.vdouble(0.119, 0.127, 0.442)
        ),
        emCaloDR = cms.double(0.1),
        maxInvisiblePt = cms.double(10.0),
        nAllNeutral = cms.uint32(25),
        nCalo = cms.uint32(18),
        nEmCalo = cms.uint32(12),
        nMu = cms.uint32(2),
        nPhoton = cms.uint32(12),
        nSelCalo = cms.uint32(18),
        nTrack = cms.uint32(25),
        tightTrackMaxInvisiblePt = cms.double(20),
        trackCaloDR = cms.double(0.15),
        trackEmDR = cms.double(0.04),
        trackMuDR = cms.double(0.2)
    ),
    puAlgo = cms.string('LinearizedPuppi'),
    puAlgoParameters = cms.PSet(
        NNVtxAssociation = cms.PSet(
            associationNetworkEtaBounds = cms.vdouble(
                0.0, 0.01984126984126984, 0.03968253968253968, 0.05952380952380952, 0.07936507936507936,
                0.0992063492063492, 0.11904761904761904, 0.1388888888888889, 0.15873015873015872, 0.17857142857142855,
                0.1984126984126984, 0.21825396825396826, 0.23809523809523808, 0.2579365079365079, 0.2777777777777778,
                0.2976190476190476, 0.31746031746031744, 0.33730158730158727, 0.3571428571428571, 0.376984126984127,
                0.3968253968253968, 0.41666666666666663, 0.4365079365079365, 0.45634920634920634, 0.47619047619047616,
                0.496031746031746, 0.5158730158730158, 0.5357142857142857, 0.5555555555555556, 0.5753968253968254,
                0.5952380952380952, 0.615079365079365, 0.6349206349206349, 0.6547619047619048, 0.6746031746031745,
                0.6944444444444444, 0.7142857142857142, 0.7341269841269841, 0.753968253968254, 0.7738095238095237,
                0.7936507936507936, 0.8134920634920635, 0.8333333333333333, 0.8531746031746031, 0.873015873015873,
                0.8928571428571428, 0.9126984126984127, 0.9325396825396824, 0.9523809523809523, 0.9722222222222222,
                0.992063492063492, 1.0119047619047619, 1.0317460317460316, 1.0515873015873016, 1.0714285714285714,
                1.0912698412698412, 1.1111111111111112, 1.130952380952381, 1.1507936507936507, 1.1706349206349205,
                1.1904761904761905, 1.2103174603174602, 1.23015873015873, 1.25, 1.2698412698412698,
                1.2896825396825395, 1.3095238095238095, 1.3293650793650793, 1.349206349206349, 1.369047619047619,
                1.3888888888888888, 1.4087301587301586, 1.4285714285714284, 1.4484126984126984, 1.4682539682539681,
                1.488095238095238, 1.507936507936508, 1.5277777777777777, 1.5476190476190474, 1.5674603174603174,
                1.5873015873015872, 1.607142857142857, 1.626984126984127, 1.6468253968253967, 1.6666666666666665,
                1.6865079365079365, 1.7063492063492063, 1.726190476190476, 1.746031746031746, 1.7658730158730158,
                1.7857142857142856, 1.8055555555555554, 1.8253968253968254, 1.8452380952380951, 1.865079365079365,
                1.8849206349206349, 1.9047619047619047, 1.9246031746031744, 1.9444444444444444, 1.9642857142857142,
                1.984126984126984, 2.003968253968254, 2.0238095238095237, 2.0436507936507935, 2.0634920634920633,
                2.083333333333333, 2.1031746031746033, 2.123015873015873, 2.142857142857143, 2.1626984126984126,
                2.1825396825396823, 2.202380952380952, 2.2222222222222223, 2.242063492063492, 2.261904761904762,
                2.2817460317460316, 2.3015873015873014, 2.321428571428571, 2.341269841269841, 2.361111111111111,
                2.380952380952381, 2.4007936507936507, 2.4206349206349205, 2.4404761904761902, 2.46031746031746,
                2.4801587301587302, 2.5
            ),
            associationNetworkPath = cms.string('L1TNNVtx_Assoc_Model_v0'),
            associationNetworkZ0ResBins = cms.vdouble(
                127.0, 126.0, 126.0, 126.0, 125.0,
                124.0, 123.0, 122.0, 120.0, 119.0,
                117.0, 115.0, 114.0, 112.0, 110.0,
                107.0, 105.0, 103.0, 101.0, 98.0,
                96.0, 94.0, 91.0, 89.0, 87.0,
                85.0, 82.0, 80.0, 78.0, 76.0,
                74.0, 72.0, 70.0, 68.0, 66.0,
                64.0, 62.0, 61.0, 59.0, 57.0,
                56.0, 54.0, 53.0, 51.0, 50.0,
                48.0, 47.0, 46.0, 45.0, 43.0,
                42.0, 41.0, 40.0, 39.0, 38.0,
                37.0, 36.0, 35.0, 34.0, 33.0,
                33.0, 32.0, 31.0, 30.0, 30.0,
                29.0, 28.0, 28.0, 27.0, 26.0,
                26.0, 25.0, 24.0, 24.0, 23.0,
                23.0, 22.0, 22.0, 21.0, 21.0,
                21.0, 20.0, 20.0, 19.0, 19.0,
                18.0, 18.0, 18.0, 17.0, 17.0,
                17.0, 16.0, 16.0, 16.0, 15.0,
                15.0, 15.0, 15.0, 14.0, 14.0,
                14.0, 14.0, 13.0, 13.0, 13.0,
                13.0, 12.0, 12.0, 12.0, 12.0,
                12.0, 11.0, 11.0, 11.0, 11.0,
                11.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 10.0, 9.0, 9.0, 9.0,
                9.0, 9.0, 0.0
            ),
            associationNetworkZ0binning = cms.vdouble(-20.46912512, 20.46912512, 0.15991504),
            associationThreshold = cms.double(-1.0)
        ),
        absEtaCuts = cms.vdouble(),
        alphaCrop = cms.vdouble(4),
        alphaSlopes = cms.vdouble(0.7),
        alphaZeros = cms.vdouble(6.0),
        dZ = cms.double(0.5),
        dr = cms.double(0.3),
        drMin = cms.double(0.07),
        finalSortAlgo = cms.string('Insertion'),
        nFinalSort = cms.uint32(18),
        nIn = cms.uint32(25),
        nOut = cms.uint32(25),
        nTrack = cms.uint32(22),
        nVtx = cms.uint32(1),
        priors = cms.vdouble(5.0),
        priorsPhoton = cms.vdouble(1.0),
        ptCut = cms.vdouble(1.0),
        ptMax = cms.double(50.0),
        ptSlopes = cms.vdouble(0.3),
        ptSlopesPhoton = cms.vdouble(0.3),
        ptZeros = cms.vdouble(4.0),
        ptZerosPhoton = cms.vdouble(2.5),
        useMLAssociation = cms.bool(False)
    ),
    rawGCTSectors = cms.VPSet(cms.PSet(
        etaBoundaries = cms.vdouble(-1.5, 1.5),
        phiSlices = cms.uint32(3),
        phiZero = cms.double(0.6981317007977318)
    )),
    regionizerAlgo = cms.string('Ideal'),
    regions = cms.VPSet(cms.PSet(
        etaBoundaries = cms.vdouble(
            -1.5, -1.0, -0.5, 0.0, 0.5,
            1.0, 1.5
        ),
        phiSlices = cms.uint32(9)
    )),
    tkEgAlgoParameters = cms.PSet(
        absEtaBoundaries = cms.vdouble(0.0, 0.9, 1.5),
        algorithm = cms.uint32(4),
        caloEtMin = cms.double(0.0),
        caloHwQual = cms.int32(4),
        compositeParametersTkEle = cms.VPSet(
            cms.PSet(
                dEta_max = cms.double(-1),
                dPhi_max = cms.double(-1),
                loose_wp = cms.PSet(
                    bins = cms.vdouble(0),
                    values = cms.vdouble(-1)
                ),
                model = cms.string(''),
                tight_wp = cms.PSet(
                    bins = cms.vdouble(0),
                    values = cms.vdouble(-1)
                )
            ),
            cms.PSet(
                dEta_max = cms.double(0.2),
                dPhi_max = cms.double(0.2),
                loose_wp = cms.PSet(
                    bins = cms.vdouble(0.0, 18.0, 28.0, 36.0),
                    values = cms.vdouble(-0.181641, -0.15, 0.075, -0.181641)
                ),
                model = cms.string('L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EE_v0.json'),
                tight_wp = cms.PSet(
                    bins = cms.vdouble(0.0, 28.0, 40.0),
                    values = cms.vdouble(0.0527344, 0.3, 0.0527344)
                )
            ),
            cms.PSet(
                dEta_max = cms.double(0.03),
                dPhi_max = cms.double(0.3),
                loose_wp = cms.PSet(
                    bins = cms.vdouble(0),
                    values = cms.vdouble(-1)
                ),
                model = cms.string('L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EB_v0.json'),
                tight_wp = cms.PSet(
                    bins = cms.vdouble(
                        0, 5, 10, 20, 30,
                        50
                    ),
                    values = cms.vdouble(
                        0.19, 0.05, -0.35, -0.45, -0.5,
                        -0.38
                    )
                )
            ),
            cms.PSet(
                dEta_max = cms.double(0.03),
                dPhi_max = cms.double(0.3),
                loose_wp = cms.PSet(
                    bins = cms.vdouble(0),
                    values = cms.vdouble(-1)
                ),
                model = cms.string('L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EE_v1.json'),
                tight_wp = cms.PSet(
                    bins = cms.vdouble(
                        0, 5, 10, 20, 30,
                        50
                    ),
                    values = cms.vdouble(
                        0.309191, 0.0909913, -0.211824, -0.223846, -0.188443,
                        -0.15476
                    )
                )
            ),
            cms.PSet(
                dEta_max = cms.double(0.03),
                dPhi_max = cms.double(0.3),
                loose_wp = cms.PSet(
                    bins = cms.vdouble(0),
                    values = cms.vdouble(-1)
                ),
                model = cms.string('L1Trigger/Phase2L1ParticleFlow/data/egamma/compositeID_EB_v1.1.json'),
                tight_wp = cms.PSet(
                    bins = cms.vdouble(
                        0, 5.75, 9.25, 11.25, 15,
                        28.75, 33.25
                    ),
                    values = cms.vdouble(
                        0.09375, -0.015625, -0.0742188, -0.109375, -0.144531,
                        -0.201172, -0.242188
                    )
                )
            )
        ),
        dEtaMaxBrem = cms.double(0.02),
        dEtaValues = cms.vdouble(0.025, 0.015, 0.01),
        dPhiMaxBrem = cms.double(0.1),
        dPhiValues = cms.vdouble(0.07, 0.07, 0.07),
        doBremRecovery = cms.bool(False),
        doEndcapHwQual = cms.bool(False),
        doPfIso = cms.bool(True),
        doTkIso = cms.bool(True),
        filterHwQuality = cms.bool(False),
        hwIsoTypeTkEle = cms.uint32(0),
        hwIsoTypeTkEm = cms.uint32(0),
        nCompCandPerCluster = cms.uint32(3),
        nEMCALO_EGIN = cms.uint32(10),
        nEM_EGOUT = cms.uint32(10),
        nTRACK = cms.uint32(25),
        nTRACK_EGIN = cms.uint32(13),
        pfIsoParametersTkEle = cms.PSet(
            dRMax = cms.double(0.2),
            dRMin = cms.double(0.03),
            dZ = cms.double(0.6),
            tkQualityPtMin = cms.double(1.0)
        ),
        pfIsoParametersTkEm = cms.PSet(
            dRMax = cms.double(0.3),
            dRMin = cms.double(0.07),
            dZ = cms.double(0.6),
            tkQualityPtMin = cms.double(1.0)
        ),
        tkIsoParametersTkEle = cms.PSet(
            dRMax = cms.double(0.2),
            dRMin = cms.double(0.03),
            dZ = cms.double(0.6),
            tkQualityPtMin = cms.double(2.0)
        ),
        tkIsoParametersTkEm = cms.PSet(
            dRMax = cms.double(0.3),
            dRMin = cms.double(0.07),
            dZ = cms.double(0.6),
            tkQualityPtMin = cms.double(2.0)
        ),
        trkQualityPtMin = cms.double(0.0),
        writeBeforeBremRecovery = cms.bool(True),
        writeEGSta = cms.bool(False)
    ),
    tkEgSorterAlgo = cms.string('Barrel'),
    tkEgSorterParameters = cms.PSet(
        nObjSorted = cms.uint32(16),
        nObjToSort = cms.uint32(16)
    ),
    trackInputConversionAlgo = cms.string('Emulator'),
    trackInputConversionParameters = cms.PSet(
        dEtaBarrelBits = cms.uint32(8),
        dEtaBarrelZ0PostShift = cms.uint32(2),
        dEtaBarrelZ0PreShift = cms.uint32(2),
        dPhiBarrelBits = cms.uint32(4),
        dPhiBarrelRInvPostShift = cms.uint32(4),
        dPhiBarrelRInvPreShift = cms.uint32(4),
        etaLUTBits = cms.uint32(10),
        etaShift = cms.uint32(5),
        phiBits = cms.uint32(10),
        ptLUTBits = cms.uint32(11),
        region = cms.string('barrel'),
        slimDataFormat = cms.bool(True),
        z0Bits = cms.uint32(12)
    ),
    tracks = cms.InputTag("l1tPFTracksFromL1TracksExtended"),
    trkPtCut = cms.double(2.0),
    vtxCollection = cms.InputTag("l1tVertexFinderEmulator","L1VerticesEmulation")
)

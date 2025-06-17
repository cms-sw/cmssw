import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.0328407225),
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minHitsPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(5*512*1024)),
    maxNumberOfTuples = cms.string(str(32*1024)), 
    cellPtCut = cms.double(0.85),
    cellZ0Cut = cms.double(7.5),
    minYsizeB1 = cms.int32(25),
    minYsizeB2 = cms.int32(15),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(20),
    avgHitsPerTrack = cms.double(7.0),
    avgCellsPerHit = cms.double(6),
    avgCellsPerCell = cms.double(0.151),
    avgTracksPerCell = cms.double(0.040),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12.),
    ),
    geometry = cms.PSet(
        caDCACuts = cms.vdouble(0.15, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25),
        caThetaCuts = cms.vdouble(0.002, 0.002, 0.002, 0.002, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003),
        startingPairs = cms.vuint32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32),
        pairGraph = cms.vuint32(0, 1, 0, 4, 0, 16, 1, 2, 1, 4, 1, 16, 2, 3, 2, 4, 2, 16, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 0, 2, 0, 5, 0, 17, 0, 6, 0, 18, 1, 3, 1, 5, 1, 17, 1, 6, 1, 18, 11, 12, 12, 13, 13, 14, 14, 15, 23, 24, 24, 25, 25, 26, 26, 27, 4, 6, 5, 7, 6, 8, 7, 9, 8, 10, 9, 11, 10, 12, 16, 18, 17, 19, 18, 20, 19, 21, 20, 22, 21, 23, 22, 24),
        phiCuts = cms.vint32(522, 522, 522, 626, 730, 730, 626, 730, 730, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 522, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 730, 522, 522, 522, 522, 522, 522, 522, 522),
        minZ = cms.vdouble(-16, 4, -22, -17, 6, -22, -18, 11, -22, 23, 30, 39, 50, 65, 82, 109, -28, -35, -44, -55, -70, -87, -113, -16, 7, -22, 11, -22, -17, 9, -22, 13, -22, 137, 173, 199, 229, -142, -177, -203, -233, 23, 30, 39, 50, 65, 82, 109, -28, -35, -44, -55, -70, -87, -113),
        maxZ = cms.vdouble(17, 22, -4, 17, 22, -6, 18, 22, -11, 28, 35, 44, 55, 70, 87, 113, -23, -30, -39, -50, -65, -82, -109, 17, 22, -7, 22, -10, 17, 22, -9, 22, -13, 142, 177, 203, 233, -137, -173, -199, -229, 28, 35, 44, 55, 70, 87, 113, -23, -30, -39, -50, -65, -82, -109),
        maxR = cms.vdouble(5, 5, 5, 7, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 6, 5, 5, 5, 6, 5, 5, 5, 9, 9, 9, 8, 8, 8, 11, 9, 9, 9, 8, 8, 8, 11)
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

_hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2OT@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsExtendedSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.01425), # corresponds to 800 MeV in 3.8T.
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minHitsPerNtuplet = cms.uint32(5),
    maxNumberOfDoublets = cms.string(str(15*512*1024)),
    maxNumberOfTuples = cms.string(str(4*60*1024)),
    cellPtCut = cms.double(0.85), # Corresponds to 1 GeV * this cut, i.e., 850 MeV, as minimum p_t
    cellZ0Cut = cms.double(12.5), # it's half the BS width! It has nothing to do with the sample!!
    minYsizeB1 = cms.int32(20),
    minYsizeB2 = cms.int32(15),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(24),
    avgHitsPerTrack = cms.double(10.0),
    avgCellsPerHit = cms.double(25),
    avgCellsPerCell = cms.double(5),
    avgTracksPerCell = cms.double(5),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12),
    ),
    geometry = cms.PSet(
        # This cut also uses the hardCurvCut parameters inside the
        # Kernel_connect "function". This is used to cut connections that have
        # either a too low p_t or that do not intersect the BS+tolerance
        # region. Internally, this cut is compared against the circle.dca0() in
        # natural units divided by circle.curvature(), where circle is the
        # circle passing through the 3 points of the triplet under
        # investigation. Therefore the cut represent the compatibility of the
        # circle in the transverse plane and the units are meant to be cm.
        caDCACuts = cms.vdouble(
            0.15,                       #  0
            0.25,                       #  1
            0.20,                       #  2
            0.20, # End PXB             #  3
            0.25,                       #  4
            0.25,                       #  5
            0.25,                       #  6
            0.25,                       #  7
            0.25,                       #  8
            0.25,                       #  9
            0.25,                       # 10
            0.25,                       # 11
            0.25,                       # 12
            0.25,                       # 13
            0.25,                       # 14
            0.25, # End PXFWD+          # 15
            0.25,                       # 16
            0.25,                       # 17
            0.25,                       # 18
            0.25,                       # 19
            0.25,                       # 20
            0.25,                       # 21
            0.25,                       # 22
            0.25,                       # 23
            0.25,                       # 24
            0.25,                       # 25
            0.25,                       # 26
            0.25, # End PXFWD-          # 27
            0.10,                       # 28
            0.10,                       # 29
            0.10), # End of OT PinPS    # 30
        # caThetaCut is used in the areAlignedRZ function to check if two
        # sibling cell are compatible in the R-Z plane. In that same function,
        # we also use ptmin variable. The caThetaCut is assigned to the SoA of
        # the layers, and is percolated into this compatibility function via
        # the SoA itself.
        caThetaCuts = cms.vdouble(
            0.002,                      #  0
            0.002,                      #  1
            0.002,                      #  2
            0.002,                      #  3
            0.003,                      #  4
            0.003,                      #  5
            0.003,                      #  6
            0.003,                      #  7
            0.003,                      #  8
            0.003,                      #  9
            0.003,                      # 10
            0.003,                      # 11
            0.003,                      # 12
            0.003,                      # 13
            0.003,                      # 14
            0.003,                      # 15
            0.003,                      # 16
            0.003,                      # 17
            0.003,                      # 18
            0.003,                      # 19
            0.003,                      # 20
            0.003,                      # 21
            0.003,                      # 22
            0.003,                      # 23
            0.003,                      # 24
            0.003,                      # 25
            0.003,                      # 26
            0.003,                      # 27
            0.003,                      # 28
            0.003,                      # 29
            0.003),                     # 30
        startingPairs = cms.vuint32(
                0,    # PXB0-1
                1,    # PXB0-4
                2,    # PXB0-16
                3,    # PXB1-2
                4,    # PXB1-4
                5,    # PXB1-16
                6,    # PXB2-3
#                7,    # PXB2-4
#                8,    # PXB2-16
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
#                27,
#                28
#                30,
#                31,
#                32,
                ),
        pairGraph = cms.vuint32(
                0, 1,                         # 0
                0, 4,                         # 1
                0, 16,                        # 2
                1, 2,                         # 3
                1, 4,                         # 4
                1, 16,                        # 5
                2, 3,                         # 6
                2, 4,                         # 7
                2, 16,                        # 8
                4, 5,                         # 9
                5, 6,                         # 10
                6, 7,                         # 11
                7, 8,                         # 12
                8, 9,                         # 13
                9, 10,                        # 14
                10, 11,                       # 15
                16, 17,                       # 16
                17, 18,                       # 17
                18, 19,                       # 18
                19, 20,                       # 19
                20, 21,                       # 20
                21, 22,                       # 21
                22, 23,                       # 22
                0, 2,                         # 23
                0, 5,                         # 24
                0, 17,                        # 25
#               0, 6,                         # 26
#               0, 18,                        # 27
                1, 3,                         # 28
                1, 5,                         # 29
                1, 17,                        # 30
#                1, 6,                        # 31
#                1, 18, # last starting pair  # 32
                11, 12,                       # 33
                12, 13,                       # 34
                13, 14,                       # 35
                14, 15,                       # 36
                23, 24,                       # 37
                24, 25,                       # 38
                25, 26,                       # 39
                26, 27,                       # 40
                4, 6,                         # 41
                5, 7,                         # 42
                6, 8,                         # 43
                7, 9,                         # 44
                8, 10,                        # 45
                9, 11,                        # 46
                10, 12,                       # 47
                16, 18,                       # 48
                17, 19,                       # 49
                18, 20,                       # 50
                19, 21,                       # 51
                20, 22,                       # 52
                21, 23,                       # 53
                22, 24,                       # 54
                 2, 28,                       # 55
                 3, 28,                       # 56
#                3, 29,                       # 57
                28, 29,                       # 58
#                28, 30,                      # 59
                29, 30,                       # 60
                 4, 28,                       # 61
                 5, 28,                       # 62
                 6, 28,                       # 63
                 7, 28,                       # 64
#                8, 28,                       # 65
#                9, 28,                       # 66
                16, 28,                       # 67
                17, 28,                       # 68
                18, 28,                       # 69
                19, 28,                       # 70
#                20, 28,                      # 71
#                21, 28,                      # 72
#                4, 29,                       # 73
#                5, 29,                       # 74
#                6, 29,                       # 75
#                7, 29,                       # 76
#                8, 29,                       # 77
#                16, 29,                      # 78
#                17, 29,                      # 79
#                18, 29,                      # 80
#                19, 29,                      # 81
#                20, 29,                      # 82
                11, 13,                       # 83
                11, 14,                       # 84
                11, 15,                       # 85
                23, 25,                       # 86
                23, 26,                       # 87
                23, 27,                       # 88
#                 1, 28,                      # 89
#                 1, 28,                      # 90
                 ),
        phiCuts = cms.vint32(
                522,   # 0
                650,   # 1
                650,   # 2
                626,   # 3
                730,   # 4
                730,   # 5
                626,   # 6
                730,   # 7
                730,   # 8
                522,   # 9
                522,   # 10
                522,   # 11
                522,   # 12
                522,   # 13
                522,   # 14
                522,   # 15
                522,   # 16
                522,   # 17
                522,   # 18
                522,   # 19
                522,   # 20
                522,   # 21
                522,   # 22
                600,   # 23
                522,   # 24
                522,   # 25
#               522,   # 26
#               522,   # 27
                650,   # 28
                730,   # 29
                730,   # 30
#               730,   # 31
#               730,   # 32
                730,   # 33
                730,   # 34
                730,   # 35
                730,   # 36
                730,   # 37
                730,   # 38
                730,   # 39
                730,   # 40
                730,   # 41
                730,   # 42
                730,   # 43
                730,   # 44
                730,   # 45
                730,   # 46
                650,   # 47
                522,   # 48
                522,   # 49
                522,   # 50
                522,   # 51
                522,   # 52
                522,   # 53
                650,   # 54
               1200,   # 55
               1000,   # 56
#              1500,   # 57
               1100,   # 58
#              2000,   # 59
               1250,   # 60
               1000,   # 61
               1000,   # 62
               1000,   # 63
               1000,   # 64
#               1000,  # 65
#               1000,  # 66
               1000,   # 67
               1000,   # 68
               1000,   # 69
               1000,   # 70
#               1000,  # 71
#               1000,  # 72
#               1000,  # 73
#               1000,  # 74
#               1000,  # 75
#               1000,  # 76
#               1000,  # 77
#               1000,  # 78
#               1000,  # 79
#               1000,  # 80
#               1000,  # 81
#               1000,  # 82
                500,   # 83
                300,   # 84
                400,   # 85
                500,   # 86
                300,   # 87
                400,   # 88
#               1300,  # 89
#               1300,  # 90
                ),
        # minZ and maxZ are the limits in Z for the inner cell of a doublets in
        # order to be able to make a doublet with the other layer.
        minZ = cms.vdouble(
              -20.0,     # 0
              4.0,       # 1
              -22.0,     # 2
              -17.0,     # 3
              6.0,       # 4
              -22.0,     # 5
              -18.0,     # 6
              11.0,      # 7
              -22.0,     # 8
              23.0,      # 9
              30.0,      # 10
              39.0,      # 11
              50.0,      # 12
              65.0,      # 13
              82.0,      # 14
              109.0,     # 15
              -28.0,     # 16
              -35.0,     # 17
              -44.0,     # 18
              -55.0,     # 19
              -70.0,     # 20
              -87.0,     # 21
              -113.0,    # 22
              -16.0,     # 23
              7.0,       # 24
              -22.0,     # 25
#             11.0,      # 26
#             -22.0,     # 27
              -17.0,     # 28
               9.0,      # 29
              -22.0,     # 30
#              13.0,     # 31
#              -22.0,    # 32
              137.0,     # 33
              173.0,     # 34
              199.0,     # 35
              229.0,     # 36
              -142.0,    # 37
              -177.0,    # 38
              -203.0,    # 39
              -233.0,    # 40
              23.0,      # 41
              30.0,      # 42
              39.0,      # 43
              50.0,      # 44
              65.0,      # 45
              82.0,      # 46
              109.0,     # 47
              -28.0,     # 48
              -35.0,     # 49
              -44.0,     # 50
              -55.0,     # 51
              -70.0,     # 52
              -87.0,     # 53
              -113.0,    # 54
                -20,     # 55
                -20,     # 56
#               -40,     # 57
                -1200,   # 58
#                -40,    # 59
                -1200,   # 60
                 23,     # 61
                 30,     # 62
                39,      # 63
                50,      # 64
#               -1000,   # 65
#               -1000,   # 66
                -28,     # 67
                -35,     # 68
                -44,     # 69
                -55,     # 70
#               -1000,   # 71
#               -1000,   # 72
#               -1000,   # 73
#               -1000,   # 74
#               -1000,   # 75
#               -1000,   # 76
#               -1000,   # 77
#               -1000,   # 78
#               -1000,   # 79
#               -1000,   # 80
#               -1000,   # 81
#               -1000,   # 82
                -1000,   # 83 
                -1000,   # 84
                -1000,   # 85
                -1000,   # 86
                -1000,   # 87
                -1000,   # 88
#               -1000,   # 89
#                15.0,   # 90
                 ),
        maxZ = cms.vdouble(
              20.0,      # 0
              22.0,      # 1
              -4.0,      # 2
              17.0,      # 3
              22.0,      # 4
              -6.0,      # 5
              18.0,      # 6
              22.0,      # 7
              -11.0,     # 8
              28.0,      # 9
              35.0,      # 10
              44.0,      # 11
              55.0,      # 12
              70.0,      # 13
              87.0,      # 14
              113.0,     # 15
              -23.0,     # 16
              -30.0,     # 17
              -39.0,     # 18
              -50.0,     # 19
              -65.0,     # 20
              -82.0,     # 21
              -109.0,    # 22
              17.0,      # 23
              22.0,      # 24
              -7.0,      # 25
#             22.0,      # 26
#             -10.0,     # 27
              17.0,      # 28
               22.0,     # 29
              -9.0,      # 30
#              22.0,     # 31
#              -13.0,    # 32
              142.0,     # 33
              177.0,     # 34
              203.0,     # 35
              233.0,     # 36
              -137.0,    # 37
              -173.0,    # 38
              -199.0,    # 39
              -229.0,    # 40
              28.0,      # 41
              35.0,      # 42
              44.0,      # 43
              55.0,      # 44
              70.0,      # 45
              87.0,      # 46
              113.0,     # 47
              -23.0,     # 48
              -30.0,     # 49
              -39.0,     # 50
              -50.0,     # 51
              -65.0,     # 52
              -82.0,     # 53
              -109.0,    # 54
                20,      # 55
                20,      # 56
#                 40,    # 57
                1200,    # 58
#                 40,    # 59
                1200,    # 60
                  28,    # 61
                35,      # 62
                44,      # 63
               55,       # 64
#                1000,   # 65
#                1000,   # 66
                -23,     # 67
                -30,     # 68
                -39,     # 69
                -50,     # 70
#                1000,   # 71
#                1000,   # 72
#                1000,   # 73
#                1000,   # 74
#                1000,   # 75
#                1000,   # 76
#                1000,   # 77
#                1000,   # 78
#                1000,   # 79
#                1000,   # 80
#                1000,   # 81
#                1000,   # 82
                 1000,   # 83 
                 1000,   # 84
                 1000,   # 85
                 1000,   # 86
                 1000,   # 87
                 1000,   # 88
#               -15.0,   # 89
#                1000,   # 90
                 ),
        maxR = cms.vdouble(
              5.0,   # 0
              10.0,  # 1
              10.0,  # 2
              7.0,   # 3
              8.0,   # 4
              8.0,   # 5
              7.0,   # 6
              7.0,   # 7
              7.0,   # 8
              6.0,   # 9
              6.0,   # 10
              6.0,   # 11
              6.0,   # 12
              5.0,   # 13
              6.0,   # 14
              5.0,   # 15
              6.0,   # 16
              6.0,   # 17
              6.0,   # 18
              6.0,   # 19
              5.0,   # 20
              6.0,   # 21
              5.0,   # 22
              10.0,  # 23
              5.0,   # 24
              5.0,   # 25
#             5.0,   # 26
#             5.0,   # 27
              10.0,  # 28
              10.0,  # 29
              10.0,  # 30
#             8.0,   # 31
#             8.0,   # 32
              6.0,   # 33
              5.0,   # 34
              5.0,   # 35
              5.0,   # 36
              6.0,   # 37
              5.0,   # 38
              5.0,   # 39
              5.0,   # 40
              9.0,   # 41
              9.0,   # 42
              9.0,   # 43
              8.0,   # 44
              8.0,   # 45
              8.0,   # 46
              11.0,  # 47
              9.0,   # 48
              9.0,   # 49
              9.0,   # 50
              8.0,   # 51
              8.0,   # 52
              8.0,   # 53
              11.0,  # 54
                60,  # 55
                60,  # 56
#               60,  # 57
                60,  # 58
#               60,  # 59
                60,  # 60
                60,  # 61
                60,  # 62
                60,  # 63
                60,  # 64
#               60,  # 65
#               60,  # 66
                60,  # 67
                60,  # 68
                60,  # 69
                60,  # 70
#               60,  # 71
#                60, # 72
#                60, # 73
#                60, # 74
#                60, # 75
#                60, # 76
#                60, # 77
#                60, # 78
#                60, # 79
#                60, # 80
#                60, # 81
#                60  # 82
                  8, # 83 
                  3, # 84
                  5, # 85
                  8, # 86
                  3, # 87
                  5, # 88
#                60, # 89
#                60, # 90
                 )
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)


def _exclude_OT_layers(hltPhase2PixelTracksSoA, layers_to_exclude = [28, 29, 30]):
    keep_indices = []
    num_pairs = len(hltPhase2PixelTracksSoA.geometry.pairGraph) // 2
    for i in range(num_pairs):
        a = hltPhase2PixelTracksSoA.geometry.pairGraph[2*i]
        b = hltPhase2PixelTracksSoA.geometry.pairGraph[2*i + 1]
        if a not in layers_to_exclude and b not in layers_to_exclude:
            keep_indices.append(i)
    # Now update in place
    # For pairGraph, build the new flat list from kept pairs
    new_pairGraph = []
    for i in keep_indices:
        new_pairGraph.extend([hltPhase2PixelTracksSoA.geometry.pairGraph[2*i], hltPhase2PixelTracksSoA.geometry.pairGraph[2*i+1]])

    hltPhase2PixelTracksSoA.geometry.pairGraph[:] = new_pairGraph
    # Update all other lists in place
    hltPhase2PixelTracksSoA.geometry.phiCuts[:] = [hltPhase2PixelTracksSoA.geometry.phiCuts[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minZ[:] = [hltPhase2PixelTracksSoA.geometry.minZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxZ[:] = [hltPhase2PixelTracksSoA.geometry.maxZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxR[:] = [hltPhase2PixelTracksSoA.geometry.maxR[i] for i in keep_indices]

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoA)

print("Using {} pair connections: {}".format(len(hltPhase2PixelTracksSoA.geometry.pairGraph), hltPhase2PixelTracksSoA.geometry.pairGraph))

import FWCore.ParameterSet.Config as cms

removeOT = False

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2OT@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsExtendedSoA'),
    ptmin = cms.double(0.9),
#    hardCurvCut = cms.double(0.0328407225),
    hardCurvCut = cms.double(0.01425), # corresponds to 800 MeV in 3.8T.
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(True),
    minHitsPerNtuplet = cms.uint32(5),
    maxNumberOfDoublets = cms.string(str(15*512*1024)),
    maxNumberOfTuples = cms.string(str(2*60*1024)),
    cellPtCut = cms.double(0.85), # Corresponds to 1 GeV * this cut, i.e., 850 MeV, as minimum p_t
    cellZ0Cut = cms.double(12.5), # it's half the BS width! It has nothing to do with the sample!!
    minYsizeB1 = cms.int32(25),
    minYsizeB2 = cms.int32(15),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(20),
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
        maxTip  = cms.double(2.5),
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
            0.25), # End of OT PinPS    # 30
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
            0.003,                      # 26
            0.003,                      # 27
            0.003,                      # 29
            0.003),                     # 30
        startingPairs = cms.vint32(
                0,    # PXB0-1
                1,    # PXB0-4
                2,    # PXB0-16
                3,    # PXB1-2
                4,    # PXB1-4
                5,    # PXB1-16
                6,    # PXB2-3
                7,    # PXB2-4
                8,    # PXB2-16
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
                27,
                28),
#                30,
#                31),
#                29,
#                30,
#                31,
#                32),
        pairGraph = cms.vint32(
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
                0, 6,                         # 26
                0, 18,                        # 27
                1, 3,                         # 28
#                2, 5,                        # 29
                1, 5,                         # 30
                1, 17,                        # 31
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
                2, 28,                        # 55
                3, 28,                        # 56
#                3, 29,                       # 57
                28, 29,                       # 58
#                28, 30,                      # 59
                29, 30,                       # 60
                 4, 28,                       # 61
                 5, 28,                       # 62
                 6, 28,                       # 63
#                7, 28,   # 64 from top 0     # 64
#                8, 28,                       # 65
#                9, 28,                       # 66
                16, 28,                       # 67
                17, 28,                       # 68
                18, 28,                       # 69
#               19, 28,                       # 70
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
#                20, 29                       # 82
                 ),
        phiCuts = cms.vint32(
                522,   # 0
                522,   # 1
                522,   # 2
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
                522,   # 23
                522,   # 24
                522,   # 25
                522,   # 26
                522,   # 27
                522,   # 28
#                730,   # 29
                730,   # 30
                730,   # 31
#                730,   # 32
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
               1000,   # 47
               1000,   # 48
               1000,   # 49
               1000,   # 50
               1000,   # 51
               1000,   # 52
               1000,   # 53
               1000,   # 54
               1000,   # 55
               1000,   # 56
#               1000,  # 57
               1000,   # 58
#               1000,  # 59
               1000,   # 60
               1000,   # 61
               1000,   # 62
               1000, # 63
#               1000,  # 64 # 64 from top 0
#               1000,  # 65
#               1000,  # 66
               1000,   # 67
               1000,   # 68
               1000, # 69
#               1000,  # 70
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
#               1000   # 82
                ),
        # minZ and maxZ are the limits in Z for the inner cell of a doublets in
        # order to be able to make a doublet with the other layer.
        minZ = cms.vdouble(
                -19,     # 0
                4,       # 1
                -22,     # 2
                -18,     # 3
                8,       # 4
                -22,     # 5
                -22,     # 6
                11,      # 7
                -22,     # 8
                23,      # 9
                30,      # 10
                39,      # 11
                50,      # 12
                65,      # 13
                82,      # 14
                109,     # 15
                -28,     # 16
                -35,     # 17
                -44,     # 18
                -55,     # 19
                -70,     # 20
                -87,     # 21
                -113,    # 22
                -19,     # 23
                7,       # 24
                -22,     # 25
                11,      # 26
                -22,     # 27
                -19,     # 28
#                9,      # 29
                7,       # 30
                -22,     # 31
#                -22,    # 32
                137,     # 33
                173,     # 34
                199,     # 35
                229,     # 36
                -142,    # 37
                -177,    # 38
                -203,    # 39
                -233,    # 40
                23,      # 41
                30,      # 42
                39,      # 43
                50,      # 44
                65,      # 45
                82,      # 46
                109,     # 47
                -28,     # 48
                -35,     # 49
                -44,     # 50
                -55,     # 51
                -70,     # 52
                -87,     # 53
                -113,    # 54
                -20,     # 55
                -20,     # 56
#                -40,    # 57
                -1200,   # 58
#                -40,    # 59
                -1200,   # 60
                 23,     # 61
                 30,     # 62
                39,      # 63
#                50,      # 64 #  64 from top 0
#                -1000,  # 65
#                -1000,  # 66
                -28,     # 67
                -35,     # 68
                -44,     # 69
#                -55,     # 70
#                -1000,  # 71
#                -1000,  # 72
#                -1000,  # 73
#                -1000,  # 74
#                -1000,  # 75
#                -1000,  # 76
#                -1000,  # 77
#                -1000,  # 78
#                -1000,  # 79
#                -1000,  # 80
#                -1000,  # 81
#                -1000   # 82
                 ),
        maxZ = cms.vdouble(
                19,      # 0
                22,      # 1
                -4,      # 2
                18,      # 3
                22,      # 4
                -8,      # 5
                22,      # 6
                22,      # 7
                -11,     # 8
                28,      # 9
                35,      # 10
                44,      # 11
                55,      # 12
                70,      # 13
                87,      # 14
                113,     # 15
                -23,     # 16
                -30,     # 17
                -39,     # 18
                -50,     # 19
                -65,     # 20
                -82,     # 21
                -109,    # 22
                19,      # 23
                22,      # 24
                -7,      # 25
                22,      # 26
                -11,     # 27
                19,      # 28
#                22,     # 29
                22,      # 30
                -7,      # 31
#                -13,    # 32
                142,     # 33
                177,     # 34
                203,     # 35
                233,     # 36
                -137,    # 37
                -173,    # 38
                -199,    # 39
                -229,    # 40
                28,      # 41
                35,      # 42
                44,      # 43
                55,      # 44
                70,      # 45
                87,      # 46
                113,     # 47
                -23,     # 48
                -30,     # 49
                -39,     # 50
                -50,     # 51
                -65,     # 52
                -82,     # 53
                -109,    # 54
                20,      # 55
                20,      # 56
#                40,     # 57
                1200,    # 58
#                40,     # 59
                1200,    # 60
                  28,    # 61
                35,      # 62
                44,      # 63
#                55,     # 64 64 gtom top 0
#                1000,   # 65
#                1000,   # 66
                -23,     # 67
                -30,     # 68
                -39,    # 69
#                -50,   # 70
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
#                1000    # 82
                 ),
        maxR = cms.vdouble(
                5,   # 0
                5,   # 1
                5,   # 2
                7,   # 3
                8,   # 4
                8,   # 5
                7,   # 6
                7,   # 7
                7,   # 8
                6,   # 9
                6,   # 10
                6,   # 11
                6,   # 12
                5,   # 13
                6,   # 14
                5,   # 15
                6,   # 16
                6,   # 17
                6,   # 18
                6,   # 19
                5,   # 20
                6,   # 21
                5,   # 22
               10,   # 23
               10,   # 24
               10,   # 25
                5,   # 26
                5,   # 27
               10,   # 28
#               10,   # 29
               10,   # 30
                8,   # 31
#                8,   # 32
                6,   # 33
                5,   # 34
                5,   # 35
                5,   # 36
                6,   # 37
                5,   # 38
                5,   # 39
                5,   # 40
                9,   # 41
                9,   # 42
                9,   # 43
                8,   # 44
                8,   # 45
                8,   # 46
                11,  # 47
                9,   # 48
                9,   # 49
                9,   # 50
                8,   # 51
                8,   # 52
                8,   # 53
                11,  # 54
                60,  # 55
                60,  # 56
#                60, # 57
                60,  # 58
#                60, # 59
                60,  # 60
                60,  # 61
                60,  # 62
                60,  # 63
#                60,  # 64 64 from top 0
#                60, # 65
#                60, # 66
                60,  # 67
                60,  # 68
                60,# 69
#                60, # 70
#                60, # 71
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
                 )
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)


def exclude_layers(hltPhase2PixelTracksSoA, layers_to_exclude):
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



if removeOT:
    ot_layers_ = [28, 29, 30]
    exclude_layers(hltPhase2PixelTracksSoA, layers_to_exclude=ot_layers_)

print("Using {} pair connections: {}".format(len(hltPhase2PixelTracksSoA.geometry.pairGraph), hltPhase2PixelTracksSoA.geometry.pairGraph))


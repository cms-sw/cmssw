import FWCore.ParameterSet.Config as cms


# list of layers to exclude from the CA (empty list doesn't exclude; [28, 29, 30] excludes the OT)
layersToExclude = []

# layers
layers = [
    #     0,        1,     2,       3
    # index, isBarrel, caDCA, caTheta
    [     0,     True,  0.15,   0.002],
    [     1,     True,  0.25,   0.002],
    [     2,     True,  0.20,   0.002],
    [     3,     True,  0.20,   0.002],
    [     4,    False,  0.25,   0.003],
    [     5,    False,  0.25,   0.003],
    [     6,    False,  0.25,   0.003],
    [     7,    False,  0.25,   0.003],
    [     8,    False,  0.25,   0.003],
    [     9,    False,  0.25,   0.003],
    [    10,    False,  0.25,   0.003],
    [    11,    False,  0.25,   0.003],
    [    12,    False,  0.25,   0.003],
    [    13,    False,  0.25,   0.003],
    [    14,    False,  0.25,   0.003],
    [    15,    False,  0.25,   0.003],
    [    16,    False,  0.25,   0.003],
    [    17,    False,  0.25,   0.003],
    [    18,    False,  0.25,   0.003],
    [    19,    False,  0.25,   0.003],
    [    20,    False,  0.25,   0.003],
    [    21,    False,  0.25,   0.003],
    [    22,    False,  0.25,   0.003],
    [    23,    False,  0.25,   0.003],
    [    24,    False,  0.25,   0.003],
    [    25,    False,  0.25,   0.003],
    [    26,    False,  0.25,   0.003],
    [    27,    False,  0.25,   0.003],
    [    28,     True,  1.00,   0.003],
    [    29,     True,  1.00,   0.003],
    [    30,     True,  1.00,   0.004],
    [    31,    False,  1.00,   0.010],
    [    32,    False,  1.00,   0.009],
    [    33,    False,  1.00,   0.009],
    [    34,    False,  1.00,   0.009],
    [    35,    False,  1.00,   0.009],
    [    36,    False,  1.00,   0.010],
    [    37,    False,  1.00,   0.009],
    [    38,    False,  1.00,   0.009],
    [    39,    False,  1.00,   0.009],
    [    40,    False,  1.00,   0.009],
]

# layerPairs for doublet building including pair-specific cut values
layerPairs = [
    #  0,  1,     2,      3,      4,      5,       6,       7,     8,      9,     10,     11
    #  i,  o, start, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts
    [  0,  1,  True,    350,  -17.0,   17.0,  -10000,   10000,   5.0,  -16.0,   16.0,  0.85],
    [  0,  2,  True,    600,  -14.0,   14.0,  -10000,   10000,  10.0,  -16.0,   16.0,  0.85],
    [  0,  4,  True,    450,    4.0,  10000,       0,    10.0,   8.0,    0.0,   25.0,  0.85],
    [  0,  5,  True,    522,    7.0,  10000,       0,   10000,   5.0,    0.0,   25.0,  0.85],
  # [  0,  6, False,    522,   11.0,  10000,       0,   10000,   5.0, -10000,  10000,  0.85],
    [  0, 16,  True,    450, -10000,   -4.0,       0,    10.0,   8.0,  -25.0,    0.0,  0.85],
    [  0, 17,  True,    522, -10000,   -7.0,       0,   10000,   5.0,  -25.0,    0.0,  0.85],
  # [  0, 18, False,    522, -10000,  -10.0,       0,   10000,   5.0, -10000,  10000,  0.85],
    [  1,  2,  True,    400,  -17.0,   17.0,  -10000,   10000,   7.0,  -13.0,   13.0,  0.85],
    [  1,  3, False,    650,  -15.0,   15.0,  -10000,   10000,  10.0,  -15.0,   15.0,  0.85],
    [  1,  4,  True,    500,    6.0,  10000,     6.5,   10000,   8.0,    0.0,   19.0,  0.85],
    [  1,  5, False,    730,    9.0,  10000,     6.5,   10000,  10.0,    0.0,   21.0,  0.85],
  # [  1,  6, False,    730,   13.0,  10000,     6.5,   10000,   8.0, -10000,  10000,  0.85],
    [  1, 16,  True,    500, -10000,   -6.0,     6.5,   10000,   8.0,  -19.0,    0.0,  0.85],
    [  1, 17, False,    730, -10000,   -9.0,     6.5,   10000,  10.0,  -21.0,    0.0,  0.85],
  # [  1, 18, False,    730, -10000,  -13.0,     6.5,   10000,   8.0, -10000,  10000,  0.85],
  # [  1, 28, False,   1300,    7.0,  10000,    30.0,    40.0, 10000,   19.0,   32.0,   1.0],
  # [  1, 28, False,   1300, -10000,   -7.0,   -40.0,   -30.0, 10000,  -32.0,  -19.0,   1.0],
    [  2,  3,  True,    350,  -18.0,   18.0,  -10000,   10000,   7.0,   -9.0,    9.0,  0.85],
    [  2,  4, False,    400,   11.0,  10000,    11.7,   10000,   7.0,    0.0,   13.0,  0.85],
    [  2, 16, False,    400, -10000,  -11.0,    11.7,   10000,   7.0,  -13.0,    0.0,  0.85],
    [  2, 28, False,   1200,    -10,     10,   -30.0,    30.0, 10000,  -15.0,   15.0,   2.0],
    [  2, 28, False,   1200,    -20,    -10,   -50.0,   -25.0, 10000,  -35.0,  -10.0,  0.85],
    [  2, 28, False,   1200,     10,     20,    25.0,    50.0, 10000,   10.0,   35.0,  0.85],
  # [  2, 28, False,   1200,    -20,     20,   -50.0,    50.0, 10000,  -35.0,   35.0,  0.85],
    [  3, 28, False,   1000,    -20,     20,   -45.0,    45.0, 10000,  -22.0,   22.0,  0.85],
  # [  3, 29, False,   1500,    -40,     40,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  4,  5,  True,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  4,  6, False,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [  4, 28, False,   1000,   11.6,  10000,    30.0,    57.5,  16.0,    5.0,   32.5,  0.85],
  # [  4, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  5,  6,  True,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  5,  7, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [  5, 28, False,   1000,   11.6,  10000,    40.0,    80.0,  16.0,  -10.0,   50.0,  0.85],
  # [  5, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  6,  7,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  6,  8, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [  6, 28, False,   1000,   11.6,  10000,    55.0,    95.0,  16.0,    5.0,   50.0,  0.85],
  # [  6, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  7,  8,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [  7,  9, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [  7, 28, False,   1000,   11.8,  10000,    70.0,   110.0,  16.0,   15.0,   70.0,  0.85],
  # [  7, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [  8,  9,  True,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [  8, 10, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [  8, 28, False,    850,      0,  10000,    80.0,   10000,  14.0,   25.0,   70.0,  0.85],
  # [  8, 29, False,   1000,      0,  10000,       0,   10000, 10000, -10000,  10000,  0.85],
    [  9, 10,  True,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85],
    [  9, 11, False,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85],
  # [  9, 28, False,   1000,      0,  10000,       0,   10000, 10000, -10000,  10000,  0.85],
    [ 10, 11,  True,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [ 10, 12, False,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85],
    [ 11, 12, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85],
    [ 11, 13, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 11, 14, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 11, 15, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85],
    [ 12, 13, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 13, 14, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 14, 15, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85],
    [ 16, 17,  True,    300,      0,   14.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 16, 18, False,    522,      0,   14.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [ 16, 28, False,   1000,   11.6,  10000,   -57.5,   -30.0,  16.0,  -32.5,   -5.0,  0.85],
  # [ 16, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 17, 18,  True,    300,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 17, 19, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [ 17, 28, False,   1000,   11.6,  10000,   -70.0,   -40.0,  16.0,  -50.0,  -10.0,  0.85],
  # [ 17, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 18, 19,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 18, 20, False,    522,      0,   13.0,     3.5,   10000,   9.0, -10000,  10000,  0.85],
    [ 18, 28, False,   1000,   11.6,  10000,   -95.0,   -55.0,  16.0,  -50.0,   -5.0,  0.85],
  # [ 18, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 19, 20,  True,    250,      0,   13.0,     3.5,   10000,   4.5, -10000,  10000,  0.85],
    [ 19, 21, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [ 19, 28, False,   1000,   11.8,  10000,  -110.0,   -70.0,  16.0,  -70.0,  -15.0,  0.85],
  # [ 19, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 20, 21,  True,    250,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [ 20, 22, False,    522,      0,   13.0,     3.5,   10000,   8.0, -10000,  10000,  0.85],
    [ 20, 28, False,   1000,      0,  10000,  -10000,   -80.0,  14.0,  -70.0,  -25.0,  0.85],
  # [ 20, 29, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 21, 22,  True,    300,      0,   13.0,     4.0,   10000,   4.5, -10000,  10000,  0.85],
    [ 21, 23, False,    522,      0,   13.0,     4.0,   10000,   8.0, -10000,  10000,  0.85],
  # [ 21, 28, False,   1000,      0,  10000,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 22, 23,  True,    240,      0,   13.0,     3.5,   10000,   4.0, -10000,  10000,  0.85],
    [ 22, 24, False,    650,   12.5,   16.5,    20.0,   10000,  10.0, -10000,  10000,  0.85],
    [ 23, 24, False,    300,      0,   16.5,     6.0,    21.0,   5.0, -10000,  10000,  0.85],
    [ 23, 25, False,    200,      0,    6.0,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 23, 26, False,    220,      0,    4.6,       0,     7.5,   3.0, -10000,  10000,  0.85],
    [ 23, 27, False,    250,      0,    6.0,       0,   10000,   4.0, -10000,  10000,  0.85],
    [ 24, 25, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 25, 26, False,    250,      0,   22.5,     7.0,   10000,   4.0, -10000,  10000,  0.85],
    [ 26, 27, False,    250,      0,   22.5,     7.0,   10000,   3.5, -10000,  10000,  0.85],
# TOB  
    [ 28, 29, False,   1100,  -1200,   1200,  -10000,   10000, 10000,  -50.0,   50.0,  0.85],
  # [ 28, 30, False,   2000,    -40,     40,  -10000,   10000, 10000, -10000,  10000,  0.85],
    [ 29, 30, False,   1250,  -1200,   1200,  -10000,   10000, 10000,  -40.0,   40.0,  0.85],
# forward TID
    [ 28, 31, False,   1100,   70.0,   130.,    20.0,    41.0,  18.0,    5.0,   80.0,  0.85], 
    [ 29, 31, False,   1100,   70.0,   130.,    35.0,    60.0,  24.0,    5.0,   55.0,  0.85],
    [ 30, 31, False,   1100,   100.,   130.,    55.0,    70.0,  14.0,   10.0,   30.0,  0.85],
    
    [ 31, 32, False,   1100,   23.0,   60.0,    25.0,    70.0,  14.0,   15.0,   30.0,  0.85],
    [ 32, 33, False,   1100,   25.0,   55.0,    30.0,    70.0,  14.0,   20.0,   40.0,  0.85],
    [ 33, 34, False,   1100,   30.0,   55.0,    35.0,    70.0,  12.0,   30.0,   50.0,  0.85],
    [ 34, 35, False,   1100,   30.0,   55.0,    35.0,    70.0,  12.0,   35.0,   60.0,  0.85],
    
    [  9, 31, False,    750,   10.0,   18.0,    20.0,    28.0,  10.0,   40.0,   50.0,  0.85],
    [  9, 32, False,   1000,   10.0,   18.0,    20.0,    30.0,  15.0,   65.0,   75.0,  0.85],
# backward TID
    [ 28, 36, False,   1100,  -130.,  -70.0,    20.0,    41.0,  18.0,  -80.0,   -5.0,  0.85], 
    [ 29, 36, False,   1100,  -130.,  -70.0,    35.0,    60.0,  24.0,  -55.0,   -5.0,  0.85],
    [ 30, 36, False,   1100,  -130.,  -100.,    55.0,    70.0,  14.0,  -30.0,  -10.0,  0.85],
    
    [ 36, 37, False,   1100,   23.0,   60.0,    25.0,    70.0,  14.0,  -30.0,  -15.0,  0.85],
    [ 37, 38, False,   1100,   25.0,   55.0,    30.0,    70.0,  14.0,  -40.0,  -20.0,  0.85],
    [ 38, 39, False,   1100,   30.0,   55.0,    35.0,    70.0,  12.0,  -50.0,  -30.0,  0.85],
    [ 39, 40, False,   1100,   30.0,   55.0,    35.0,    70.0,  12.0,  -60.0,  -35.0,  0.85],
    
    [ 21, 36, False,    750,   10.0,   18.0,    20.0,    28.0,  10.0,  -50.0,  -40.0,  0.85],
    [ 21, 37, False,   1000,   10.0,   18.0,    20.0,    30.0,  15.0,  -75.0,  -65.0,  0.85],
#      i,  o, start, phiCut,  minIn,  maxIn,  minOut,  maxOut, maxDR,  minDZ,  maxDZ, ptCuts
]

# find the layerPairs that contain a layer that is excluded
excludeLayerPair = [any([(lp[0] == l) or (lp[1] == l) for l in layersToExclude]) for lp in layerPairs]
excludeCAExtension = [any([(lp[0] == l) or (lp[1] == l) for l in range(28,41)]) for lp in layerPairs]

# exclude those layerPairs
layerPairsAlpaka = []
layerPairsCAExtension = []
for i, lp in enumerate(layerPairs):
    if (not excludeLayerPair[i]) and (not excludeCAExtension[i]):
        layerPairsAlpaka.append(lp)
    if not excludeLayerPair[i]:
        layerPairsCAExtension.append(lp)

# get startingPairs for Ntuplet building
startingPairsAlpaka = []
for i, lp in enumerate(layerPairsAlpaka):
    if lp[2]:
        startingPairsAlpaka.append(i)

startingPairsCAExtension = []
for i, lp in enumerate(layerPairsCAExtension):
    if lp[2]:
        startingPairsCAExtension.append(i)

hltPhase2PixelTracksSoA = cms.EDProducer('CAHitNtupletAlpakaPhase2@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2SiPixelRecHitsSoA'),
    ptmin = cms.double(0.9),
    hardCurvCut = cms.double(0.01425), # corresponds to 800 MeV in 3.8T.
    earlyFishbone = cms.bool(True),
    lateFishbone = cms.bool(False),
    fillStatistics = cms.bool(False),
    minHitsPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(6*512*1024)),
    maxNumberOfTuples = cms.string(str(60*1024)),
    cellZ0Cut = cms.double(12.5), # it's half the BS width! It has nothing to do with the sample!!
    minYsizeB1 = cms.int32(20),
    minYsizeB2 = cms.int32(18),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(24),
    avgHitsPerTrack = cms.double(7.0),
    avgCellsPerHit = cms.double(12),
    avgCellsPerCell = cms.double(0.151),
    avgTracksPerCell = cms.double(0.040),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2TripletsOrQuadruplets = cms.double(5.0),
        maxChi2Quintuplets = cms.double(5.0),
        maxChi2 = cms.double(5.0),
        minPt   = cms.double(0.9),
        maxTip  = cms.double(0.3),
        maxZip  = cms.double(12),
    ),
    geometry = cms.PSet(
        caDCACuts   = cms.vdouble([l[2] for l in layers[:28]]),
        caThetaCuts = cms.vdouble([l[3] for l in layers[:28]]),
        startingPairs = cms.vuint32(startingPairsAlpaka),
        pairGraph = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsAlpaka], [])),
        phiCuts   = cms.vint32( [lp[ 3] for lp in layerPairsAlpaka]),
        minInner  = cms.vdouble([lp[ 4] for lp in layerPairsAlpaka]),
        maxInner  = cms.vdouble([lp[ 5] for lp in layerPairsAlpaka]),
        minOuter  = cms.vdouble([lp[ 6] for lp in layerPairsAlpaka]),
        maxOuter  = cms.vdouble([lp[ 7] for lp in layerPairsAlpaka]),
        maxDR     = cms.vdouble([lp[ 8] for lp in layerPairsAlpaka]),
        minDZ     = cms.vdouble([lp[ 9] for lp in layerPairsAlpaka]),
        maxDZ     = cms.vdouble([lp[10] for lp in layerPairsAlpaka]),
        ptCuts    = cms.vdouble([lp[11] for lp in layerPairsAlpaka]),
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
    minHitsPerNtuplet = cms.uint32(4),
    maxNumberOfDoublets = cms.string(str(12*512*1024)),
    maxNumberOfTuples = cms.string(str(2*60*1024)),
    cellZ0Cut = cms.double(12.5), # it's half the BS width! It has nothing to do with the sample!!
    minYsizeB1 = cms.int32(20),
    minYsizeB2 = cms.int32(18),
    maxDYsize12 = cms.int32(12),
    maxDYsize = cms.int32(10),
    maxDYPred = cms.int32(24),
    avgHitsPerTrack = cms.double(8.0),
    avgCellsPerHit = cms.double(17),
    avgCellsPerCell = cms.double(0.5),
    avgTracksPerCell = cms.double(0.09),
    minHitsForSharingCut = cms.uint32(10),
    fitNas4 = cms.bool(False),
    useRiemannFit = cms.bool(False),
    doSharedHitCut = cms.bool(True),
    dupPassThrough = cms.bool(False),
    useSimpleTripletCleaner = cms.bool(True),
    trackQualityCuts = cms.PSet(
        maxChi2TripletsOrQuadruplets = cms.double(1.0),
        maxChi2Quintuplets = cms.double(3.0),
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
        caDCACuts   = cms.vdouble([l[2] for l in layers]),
        # caThetaCut is used in the areAlignedRZ function to check if two
        # sibling cell are compatible in the R-Z plane. In that same function,
        # we also use ptmin variable. The caThetaCut is assigned to the SoA of
        # the layers, and is percolated into this compatibility function via
        # the SoA itself.
        caThetaCuts = cms.vdouble([l[3] for l in layers]),
        startingPairs = cms.vuint32(startingPairsCAExtension),
        pairGraph = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairsCAExtension], [])),
        phiCuts   = cms.vint32( [lp[ 3] for lp in layerPairsCAExtension]),
        minInner  = cms.vdouble([lp[ 4] for lp in layerPairsCAExtension]),
        maxInner  = cms.vdouble([lp[ 5] for lp in layerPairsCAExtension]),
        minOuter  = cms.vdouble([lp[ 6] for lp in layerPairsCAExtension]),
        maxOuter  = cms.vdouble([lp[ 7] for lp in layerPairsCAExtension]),
        maxDR     = cms.vdouble([lp[ 8] for lp in layerPairsCAExtension]),
        minDZ     = cms.vdouble([lp[ 9] for lp in layerPairsCAExtension]),
        maxDZ     = cms.vdouble([lp[10] for lp in layerPairsCAExtension]),
        ptCuts    = cms.vdouble([lp[11] for lp in layerPairsCAExtension]),
    ),
    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

def _exclude_OT_layers(hltPhase2PixelTracksSoA, layers_to_exclude = range(28,41)):
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
    hltPhase2PixelTracksSoA.geometry.minInnerR[:] = [hltPhase2PixelTracksSoA.geometry.minInnerR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxInnerR[:] = [hltPhase2PixelTracksSoA.geometry.maxInnerR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minOuterR[:] = [hltPhase2PixelTracksSoA.geometry.minOuterR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxOuterR[:] = [hltPhase2PixelTracksSoA.geometry.maxOuterR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxDR[:] = [hltPhase2PixelTracksSoA.geometry.maxDR[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minInnerZ[:] = [hltPhase2PixelTracksSoA.geometry.minInnerZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxInnerZ[:] = [hltPhase2PixelTracksSoA.geometry.maxInnerZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minOuterZ[:] = [hltPhase2PixelTracksSoA.geometry.minOuterZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxOuterZ[:] = [hltPhase2PixelTracksSoA.geometry.maxOuterZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.minDZ[:] = [hltPhase2PixelTracksSoA.geometry.minDZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.maxDZ[:] = [hltPhase2PixelTracksSoA.geometry.maxDZ[i] for i in keep_indices]
    hltPhase2PixelTracksSoA.geometry.ptCuts[:] = [hltPhase2PixelTracksSoA.geometry.ptCuts[i] for i in keep_indices]

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoA)

#print("Using {} pair connections: {}".format(len(hltPhase2PixelTracksSoA.geometry.pairGraph) // 2, hltPhase2PixelTracksSoA.geometry.pairGraph))

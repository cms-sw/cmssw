import FWCore.ParameterSet.Config as cms


# list of layers to exclude from the CA (empty list doesn't exclude; [28, 29, 30] excludes the OT)
layersToExclude = []

# layerPairs for doublet building including pair-specific cut values
layerPairs = [
    #  0,  1,     2,      3,      4,      5,       6,       7,     8,      9,     10,      11,      12,    13,    14
    #  i,  o, start, phiCut, minInR, maxInR, minOutR, maxOutR, maxDR, minInZ, maxInZ, minOutZ, maxOutZ, minDZ, maxDZ
    [  0,  1,  True,    522,      0,  10000,       0,   10000,   5.0,  -20.0,   20.0,  -10000,   10000,-10000, 10000],
    [  0,  2,  True,    600,      0,  10000,       0,   10000,  10.0,  -16.0,   17.0,  -10000,   10000,-10000, 10000],
    [  0,  4,  True,    650,      0,  10000,       0,   10000,  10.0,    4.0,   22.0,  -10000,   10000,-10000, 10000],
    [  0,  5,  True,    522,      0,  10000,       0,   10000,   5.0,    7.0,   22.0,  -10000,   10000,-10000, 10000],
  # [  0,  6, False,    522,      0,  10000,       0,   10000,   5.0,   11.0,   22.0,  -10000,   10000,-10000, 10000],
    [  0, 16,  True,    650,      0,  10000,       0,   10000,  10.0,  -22.0,   -4.0,  -10000,   10000,-10000, 10000],
    [  0, 17,  True,    522,      0,  10000,       0,   10000,   5.0,  -22.0,   -7.0,  -10000,   10000,-10000, 10000],
  # [  0, 18, False,    522,      0,  10000,       0,   10000,   5.0,  -22.0,  -10.0,  -10000,   10000,-10000, 10000],
    [  1,  2,  True,    626,      0,  10000,       0,   10000,   7.0,  -17.0,   17.0,  -10000,   10000,-10000, 10000],
    [  1,  3, False,    650,      0,  10000,       0,   10000,  10.0,  -17.0,   17.0,  -10000,   10000,-10000, 10000],
    [  1,  4,  True,    730,      0,  10000,       0,   10000,   8.0,    6.0,   22.0,  -10000,   10000,-10000, 10000],
    [  1,  5, False,    730,      0,  10000,       0,   10000,  10.0,    9.0,   22.0,  -10000,   10000,-10000, 10000],
  # [  1,  6, False,    730,      0,  10000,       0,   10000,   8.0,   13.0,   22.0,  -10000,   10000,-10000, 10000],
    [  1, 16,  True,    730,      0,  10000,       0,   10000,   8.0,  -22.0,   -6.0,  -10000,   10000,-10000, 10000],
    [  1, 17, False,    730,      0,  10000,       0,   10000,  10.0,  -22.0,   -9.0,  -10000,   10000,-10000, 10000],
  # [  1, 18, False,    730,      0,  10000,       0,   10000,   8.0,  -22.0,  -13.0,  -10000,   10000,-10000, 10000],
    [  2,  3,  True,    626,      0,  10000,       0,   10000,   7.0,  -18.0,   18.0,  -10000,   10000,-10000, 10000],
    [  2,  4, False,    730,      0,  10000,       0,   10000,   7.0,   11.0,   22.0,  -10000,   10000,-10000, 10000],
    [  2, 16, False,    730,      0,  10000,       0,   10000,   7.0,  -22.0,  -11.0,  -10000,   10000,-10000, 10000],
    [  2, 28, False,   1200,      0,  10000,       0,   10000,    60,    -20,     20,  -10000,   10000,-10000, 10000],
    [  3, 28, False,   1000,      0,  10000,       0,   10000,    60,    -20,     20,  -10000,   10000,-10000, 10000],
  # [  3, 29, False,   1500,      0,  10000,       0,   10000,    60,    -40,     40,  -10000,   10000,-10000, 10000],
    [  4,  5,  True,    522,      0,  10000,       0,   10000,   6.0,   23.0,   28.0,  -10000,   10000,-10000, 10000],
    [  4,  6, False,    730,      0,  10000,       0,   10000,   9.0,   23.0,   28.0,  -10000,   10000,-10000, 10000],
    [  4, 28, False,   1000,      0,  10000,       0,   10000,    60,     23,     28,  -10000,   10000,-10000, 10000],
  # [  4, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [  5,  6,  True,    522,      0,  10000,       0,   10000,   6.0,   30.0,   35.0,  -10000,   10000,-10000, 10000],
    [  5,  7, False,    730,      0,  10000,       0,   10000,   9.0,   30.0,   35.0,  -10000,   10000,-10000, 10000],
    [  5, 28, False,   1000,      0,  10000,       0,   10000,    60,     30,     35,  -10000,   10000,-10000, 10000],
  # [  5, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [  6,  7,  True,    522,      0,  10000,       0,   10000,   6.0,   39.0,   44.0,  -10000,   10000,-10000, 10000],
    [  6,  8, False,    730,      0,  10000,       0,   10000,   9.0,   39.0,   44.0,  -10000,   10000,-10000, 10000],
    [  6, 28, False,   1000,      0,  10000,       0,   10000,    60,     39,     44,  -10000,   10000,-10000, 10000],
  # [  6, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [  7,  8,  True,    522,      0,  10000,       0,   10000,   6.0,   50.0,   55.0,  -10000,   10000,-10000, 10000],
    [  7,  9, False,    730,      0,  10000,       0,   10000,   8.0,   50.0,   55.0,  -10000,   10000,-10000, 10000],
    [  7, 28, False,   1000,      0,  10000,       0,   10000,    60,     50,     55,  -10000,   10000,-10000, 10000],
  # [  7, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [  8,  9,  True,    522,      0,  10000,       0,   10000,   5.0,   65.0,   70.0,  -10000,   10000,-10000, 10000],
    [  8, 10, False,    730,      0,  10000,       0,   10000,   8.0,   65.0,   70.0,  -10000,   10000,-10000, 10000],
  # [  8, 28, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
  # [  8, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [  9, 10,  True,    522,      0,  10000,       0,   10000,   6.0,   82.0,   87.0,  -10000,   10000,-10000, 10000],
    [  9, 11, False,    730,      0,  10000,       0,   10000,   8.0,   82.0,   87.0,  -10000,   10000,-10000, 10000],
  # [  9, 28, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 10, 11,  True,    522,      0,  10000,       0,   10000,   5.0,  109.0,  113.0,  -10000,   10000,-10000, 10000],
    [ 10, 12, False,    650,      0,  10000,       0,   10000,  11.0,  109.0,  113.0,  -10000,   10000,-10000, 10000],
    [ 11, 12, False,    730,      0,  10000,       0,   10000,   6.0,  137.0,  142.0,  -10000,   10000,-10000, 10000],
    [ 11, 13, False,    500,      0,  10000,       0,   10000,     8,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 11, 14, False,    300,      0,  10000,       0,   10000,     3,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 11, 15, False,    400,      0,  10000,       0,   10000,     5,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 12, 13, False,    730,      0,  10000,       0,   10000,   5.0,  173.0,  177.0,  -10000,   10000,-10000, 10000],
    [ 13, 14, False,    730,      0,  10000,       0,   10000,   5.0,  199.0,  203.0,  -10000,   10000,-10000, 10000],
    [ 14, 15, False,    730,      0,  10000,       0,   10000,   5.0,  229.0,  233.0,  -10000,   10000,-10000, 10000],
    [ 16, 17,  True,    522,      0,  10000,       0,   10000,   6.0,  -28.0,  -23.0,  -10000,   10000,-10000, 10000],
    [ 16, 18, False,    522,      0,  10000,       0,   10000,   9.0,  -28.0,  -23.0,  -10000,   10000,-10000, 10000],
    [ 16, 28, False,   1000,      0,  10000,       0,   10000,    60,    -28,    -23,  -10000,   10000,-10000, 10000],
  # [ 16, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 17, 18,  True,    522,      0,  10000,       0,   10000,   6.0,  -35.0,  -30.0,  -10000,   10000,-10000, 10000],
    [ 17, 19, False,    522,      0,  10000,       0,   10000,   9.0,  -35.0,  -30.0,  -10000,   10000,-10000, 10000],
    [ 17, 28, False,   1000,      0,  10000,       0,   10000,    60,    -35,    -30,  -10000,   10000,-10000, 10000],
  # [ 17, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 18, 19,  True,    522,      0,  10000,       0,   10000,   6.0,  -44.0,  -39.0,  -10000,   10000,-10000, 10000],
    [ 18, 20, False,    522,      0,  10000,       0,   10000,   9.0,  -44.0,  -39.0,  -10000,   10000,-10000, 10000],
    [ 18, 28, False,   1000,      0,  10000,       0,   10000,    60,    -44,    -39,  -10000,   10000,-10000, 10000],
  # [ 18, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 19, 20,  True,    522,      0,  10000,       0,   10000,   6.0,  -55.0,  -50.0,  -10000,   10000,-10000, 10000],
    [ 19, 21, False,    522,      0,  10000,       0,   10000,   8.0,  -55.0,  -50.0,  -10000,   10000,-10000, 10000],
    [ 19, 28, False,   1000,      0,  10000,       0,   10000,    60,    -55,    -50,  -10000,   10000,-10000, 10000],
  # [ 19, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 20, 21,  True,    522,      0,  10000,       0,   10000,   5.0,  -70.0,  -65.0,  -10000,   10000,-10000, 10000],
    [ 20, 22, False,    522,      0,  10000,       0,   10000,   8.0,  -70.0,  -65.0,  -10000,   10000,-10000, 10000],
  # [ 20, 28, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
  # [ 20, 29, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 21, 22,  True,    522,      0,  10000,       0,   10000,   6.0,  -87.0,  -82.0,  -10000,   10000,-10000, 10000],
    [ 21, 23, False,    522,      0,  10000,       0,   10000,   8.0,  -87.0,  -82.0,  -10000,   10000,-10000, 10000],
  # [ 21, 28, False,   1000,      0,  10000,       0,   10000,    60,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 22, 23,  True,    522,      0,  10000,       0,   10000,   5.0, -113.0, -109.0,  -10000,   10000,-10000, 10000],
    [ 22, 24, False,    650,      0,  10000,       0,   10000,  11.0, -113.0, -109.0,  -10000,   10000,-10000, 10000],
    [ 23, 24, False,    730,      0,  10000,       0,   10000,   6.0, -142.0, -137.0,  -10000,   10000,-10000, 10000],
    [ 23, 25, False,    500,      0,  10000,       0,   10000,     8,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 23, 26, False,    300,      0,  10000,       0,   10000,     3,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 23, 27, False,    400,      0,  10000,       0,   10000,     5,  -1000,   1000,  -10000,   10000,-10000, 10000],
    [ 24, 25, False,    730,      0,  10000,       0,   10000,   5.0, -177.0, -173.0,  -10000,   10000,-10000, 10000],
    [ 25, 26, False,    730,      0,  10000,       0,   10000,   5.0, -203.0, -199.0,  -10000,   10000,-10000, 10000],
    [ 26, 27, False,    730,      0,  10000,       0,   10000,   5.0, -233.0, -229.0,  -10000,   10000,-10000, 10000],
    [ 28, 29, False,   1100,      0,  10000,       0,   10000,    60,  -1200,   1200,  -10000,   10000,-10000, 10000],
  # [ 28, 30, False,   2000,      0,  10000,       0,   10000,    60,    -40,     40,  -10000,   10000,-10000, 10000],
    [ 29, 30, False,   1250,      0,  10000,       0,   10000,    60,  -1200,   1200,  -10000,   10000,-10000, 10000],
  # [  1, 28, False,   1300,      0,  10000,       0,   10000,    60,  -1000,  -15.0,  -10000,   10000,-10000, 10000],
  # [  1, 28, False,   1300,      0,  10000,       0,   10000,    60,   15.0,   1000,  -10000,   10000,-10000, 10000],
]

# find the layerPairs that contain a layer that is excluded
excludeLayerPair = [any([(lp[0] == l) or (lp[1] == l) for l in layersToExclude]) for lp in layerPairs]

# exclude those layerPairs
layerPairs_ = []
for i, lp in enumerate(layerPairs):
    if not excludeLayerPair[i]:
        layerPairs_.append(lp)
layerPairs = layerPairs_

# get startingPairs for Ntuplet building
startingPairs = []
for i, lp in enumerate(layerPairs):
    if lp[2]:
        startingPairs.append(i)


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
        minInnerZ = cms.vdouble(-16, 4, -22, -17, 6, -22, -18, 11, -22, 23, 30, 39, 50, 65, 82, 109, -28, -35, -44, -55, -70, -87, -113, -16, 7, -22, 11, -22, -17, 9, -22, 13, -22, 137, 173, 199, 229, -142, -177, -203, -233, 23, 30, 39, 50, 65, 82, 109, -28, -35, -44, -55, -70, -87, -113),
        maxInnerZ = cms.vdouble(17, 22, -4, 17, 22, -6, 18, 22, -11, 28, 35, 44, 55, 70, 87, 113, -23, -30, -39, -50, -65, -82, -109, 17, 22, -7, 22, -10, 17, 22, -9, 22, -13, 142, 177, 203, 233, -137, -173, -199, -229, 28, 35, 44, 55, 70, 87, 113, -23, -30, -39, -50, -65, -82, -109),
        maxDR = cms.vdouble(5, 5, 5, 7, 8, 8, 7, 7, 7, 6, 6, 6, 6, 5, 6, 5, 6, 6, 6, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 6, 5, 5, 5, 6, 5, 5, 5, 9, 9, 9, 8, 8, 8, 11, 9, 9, 9, 8, 8, 8, 11),
        minOuterZ = cms.vdouble(-10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000),
        maxOuterZ = cms.vdouble( 10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000),
        minInnerR = cms.vdouble(-10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000),
        maxInnerR = cms.vdouble( 10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000),
        minOuterR = cms.vdouble(-10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000),
        maxOuterR = cms.vdouble( 10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000),
        minDZ     = cms.vdouble(-10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000, -10000),
        maxDZ     = cms.vdouble( 10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000,  10000),
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
        startingPairs = cms.vuint32(startingPairs),
        pairGraph = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairs], [])),
        phiCuts   = cms.vint32( [lp[ 3] for lp in layerPairs]),
        minInnerR = cms.vdouble([lp[ 4] for lp in layerPairs]),
        maxInnerR = cms.vdouble([lp[ 5] for lp in layerPairs]),
        minOuterR = cms.vdouble([lp[ 6] for lp in layerPairs]),
        maxOuterR = cms.vdouble([lp[ 7] for lp in layerPairs]),
        maxDR     = cms.vdouble([lp[ 8] for lp in layerPairs]),
        minInnerZ = cms.vdouble([lp[ 9] for lp in layerPairs]),
        maxInnerZ = cms.vdouble([lp[10] for lp in layerPairs]),
        minOuterZ = cms.vdouble([lp[11] for lp in layerPairs]),
        maxOuterZ = cms.vdouble([lp[12] for lp in layerPairs]),
        minDZ     = cms.vdouble([lp[13] for lp in layerPairs]),
        maxDZ     = cms.vdouble([lp[14] for lp in layerPairs]),
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

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(hltPhase2PixelTracksSoA, _hltPhase2PixelTracksSoA)

print("Using {} pair connections: {}".format(len(hltPhase2PixelTracksSoA.geometry.pairGraph), hltPhase2PixelTracksSoA.geometry.pairGraph))

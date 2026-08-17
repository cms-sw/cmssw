import FWCore.ParameterSet.Config as cms

# CA layers of the stub-seeded graph: 28 pixel + 6 OT barrel + 20 OT disk = 54.
# Inside-out ordering: 0-27 pixel, 28-33 OT barrel, 34-43 OT disks at z > 0, 44-53 OT disks at z < 0;
# each OT disk is a PS layer (even id) followed by a 2S layer (odd id).
# Per-layer columns feed the quadruplet curvature cut (DCurv/DCurv0) and fishbone cleaning (fishCut).

layers = [
    #     0,        1,     2,         3,        4,        5
    # index, isBarrel, startR,    DCurv,   DCurv0,  fishCut
    # Pixel barrel layers
    [     0,     True,   99.0,     99.0,     99.0, 0.99999],
    [     1,     True,   99.0,     99.0,     99.0, 0.99999],
    [     2,     True,   99.0,     99.0,     99.0, 0.99999],
    [     3,     True,   99.0, 4.90e-02, 8.54e-04, 0.99999],
    # Pixel endcap layers (forward)
    [     4,    False,   99.0, 6.68e-02, 9.13e-04, 0.99999],
    [     5,    False,   99.0, 8.73e-02, 1.67e-03, 0.99999],
    [     6,    False,   99.0, 1.02e-01, 3.65e-03, 0.99999],
    [     7,    False,   99.0, 1.06e-01, 4.16e-03, 0.99999],
    [     8,    False,   99.0, 5.24e-02, 4.87e-03, 0.999999],
    [     9,    False,   99.0, 1.18e-01, 4.04e-03, 0.999999],
    [    10,    False,   99.0, 1.12e-01, 4.01e-03, 0.999999],
    [    11,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    12,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    13,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    14,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    15,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    # Pixel endcap layers (backward)
    [    16,    False,   99.0, 6.68e-02, 9.13e-04, 0.99999],
    [    17,    False,   99.0, 8.73e-02, 1.67e-03, 0.99999],
    [    18,    False,   99.0, 1.02e-01, 3.65e-03, 0.99999],
    [    19,    False,   99.0, 1.06e-01, 4.16e-03, 0.99999],
    [    20,    False,   99.0, 5.24e-02, 4.87e-03, 0.999999],
    [    21,    False,   99.0, 1.18e-01, 4.04e-03, 0.999999],
    [    22,    False,   99.0, 1.12e-01, 4.01e-03, 0.999999],
    [    23,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    24,    False,   99.0, 8.79e-02, 4.15e-03, 0.999999],
    [    25,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    26,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    [    27,    False,   99.0, 1.05e-01, 4.33e-03, 0.999999],
    # OT barrel layers 1-6 (stubs)
    [    28,     True,   99.0, 8.16e-02, 8.32e-04, 0.99999],
    [    29,     True,   99.0, 4.82e-02, 4.27e-04, 0.99999],
    [    30,     True,   99.0, 3.79e-02, 2.76e-04, 0.99999],
    [    31,     True,   99.0, 3.79e-02, 2.76e-04, 0.99999],
    [    32,     True,   99.0, 3.79e-02, 2.76e-04, 0.99999],
    [    33,     True,   99.0, 3.79e-02, 2.76e-04, 0.99999],
    # OT endcap disks 1-5 at z > 0 (stubs), PS/2S alternating
    [    34,    False,   99.0,     99.0,      99.0, 0.99999],
    [    35,    False,   99.0,     99.0,      99.0, 0.99999],
    [    36,    False,   99.0,     99.0,      99.0, 0.99999],
    [    37,    False,   99.0,     99.0,      99.0, 0.99999],
    [    38,    False,   99.0,     99.0,      99.0, 0.99999],
    [    39,    False,   99.0,     99.0,      99.0, 0.99999],
    [    40,    False,   99.0,     99.0,      99.0, 0.99999],
    [    41,    False,   99.0,     99.0,      99.0, 0.99999],
    [    42,    False,   99.0,     99.0,      99.0, 0.99999],
    [    43,    False,   99.0,     99.0,      99.0, 0.99999],
    # OT endcap disks 1-5 at z < 0 (stubs), PS/2S alternating
    [    44,    False,   99.0,     99.0,      99.0, 0.99999],
    [    45,    False,   99.0,     99.0,      99.0, 0.99999],
    [    46,    False,   99.0,     99.0,      99.0, 0.99999],
    [    47,    False,   99.0,     99.0,      99.0, 0.99999],
    [    48,    False,   99.0,     99.0,      99.0, 0.99999],
    [    49,    False,   99.0,     99.0,      99.0, 0.99999],
    [    50,    False,   99.0,     99.0,      99.0, 0.99999],
    [    51,    False,   99.0,     99.0,      99.0, 0.99999],
    [    52,    False,   99.0,     99.0,      99.0, 0.99999],
    [    53,    False,   99.0,     99.0,      99.0, 0.99999],
]

# layerPairs for doublet building, including the pair-specific cut values.
# Cut anchors: caTheta (col 17) is read at the OUTER cell's pair; caDCA (col 18)
# and floor (col 19) are read at the INNER cell's pair; geomKappa/phiMid (cols 15/16)
# read at the OUTER cell's pair. Threshold = caDCA * |curvature| + max(floor, 0).
# stubSigma (col 14) -> geometry.maxStubCurvSigma (stub-stub significance; -1 = disabled).
# z0Cuts (col 13) -> geometry.cellZ0CutPerPair (per-pair form of the scalar cellZ0Cut).
layerPairs = [
    #  0,   1,      2,      3,       4,       5,       6,        7,        8,      9,      10,      11,      12,      13,         14,         15,      16,       17,      18,       19
    #  i,   o,  start,   skip,  phiCut,   minIn,   maxIn,   minOut,   maxOut,  maxDR,   minDZ,   maxDZ,  ptCuts,  z0Cuts,  stubSigma,  geomKappa,  phiMid,  caTheta,   caDCA,    floor
    # Pixel-only connections (same as Phase2OT)
    [  0,   1,   True,  False,     350,   -17.0,    17.0,   -10000,    10000,    5.0,   -16.0,    16.0,     0.7,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  0,   2,   True,   True,     600,   -14.0,    14.0,   -10000,    10000,   10.0,   -16.0,    16.0,     0.8,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  0,   4,   True,  False,     450,     3.0,   10000,        0,     12.0,    8.5,     0.0,    25.0,     0.6,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  0,   5,   True,   True,     522,     7.0,   10000,        0,    10000,    5.0,     0.0,    25.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  0,  16,   True,  False,     450,  -10000,    -3.0,        0,     12.0,    8.5,   -25.0,     0.0,     0.6,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  0,  17,   True,   True,     522,  -10000,    -7.0,        0,    10000,    5.0,   -25.0,     0.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  1,   2,   True,  False,     400,   -17.0,    17.0,   -10000,    10000,    7.0,   -13.0,    13.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  1,   3,  False,   True,     650,   -15.0,    15.0,   -10000,    10000,   10.0,   -17.0,    17.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,     0.2,      0.0],
    [  1,   4,   True,  False,     500,     6.0,   10000,      6.5,    10000,    8.5,     0.0,    19.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  1,   5,  False,   True,     730,     9.0,   10000,      6.5,    10000,   10.0,     0.0,    21.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  1,  16,   True,  False,     500,  -10000,    -6.0,      6.5,    10000,    8.5,   -19.0,     0.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  1,  17,  False,   True,     730,  -10000,    -9.0,      6.5,    10000,   10.0,   -21.0,     0.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  2,   3,  False,  False,     350,   -18.0,    18.0,   -10000,    10000,    7.0,    -9.0,     9.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,     0.2,      0.0],
    [  2,   4,  False,  False,     400,    11.0,   10000,     11.7,    10000,    7.0,     0.0,    13.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    [  2,  16,  False,  False,     400,  -10000,   -11.0,     11.7,    10000,    7.0,   -13.0,     0.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,    0.25,      0.0],
    # Pixel barrel L3 (CA 2) to OT barrel L1 (CA 28)
    [  2,  28,  False,   True,    1200,     -10,      10,    -30.0,     30.0,  10000,   -15.0,    15.0,     2.0,    13.0,        4.0,       -1.0,    -1.0,    0.002,     0.5,      0.0],  # central
    [  2,  28,  False,  False,    1200,     -20,     -10,    -50.0,    -25.0,  10000,   -35.0,   -10.0,    0.85,    13.0,        4.0,       -1.0,    -1.0,    0.002,     0.5,      0.0],  # backward
    [  2,  28,  False,  False,    1200,      10,      20,     25.0,     50.0,  10000,    10.0,    35.0,    0.85,    13.0,        4.0,       -1.0,    -1.0,    0.002,     0.5,      0.0],  # forward
    # Pixel barrel L4 (CA 3) to OT barrel L1 (CA 28)
    [  3,  28,  False,  False,    1000,     -20,      20,    -45.0,     45.0,  10000,   -22.0,    22.0,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.002,     0.5,      0.0],
    # Pixel forward endcap disks to OT barrel L1 (CA 28)
    [  4,  28,  False,  False,    1000,    11.6,   10000,     30.0,     57.5,   16.0,     5.0,    32.5,    0.85,    13.0,        5.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [  5,  28,  False,  False,    1000,    11.6,   10000,     40.0,     70.0,   16.0,     5.0,    50.0,    0.85,    13.0,        5.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [  6,  28,  False,  False,    1000,    11.6,   10000,     55.0,     95.0,   16.0,     5.0,    50.0,    0.85,    13.0,        8.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [  7,  28,  False,  False,    1000,    11.8,   10000,     70.0,    110.0,   16.0,    15.0,    70.0,    0.85,    13.0,        8.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [  8,  28,  False,  False,    1000,       0,   10000,     80.0,    10000,   14.0,    25.0,    70.0,    0.85,    13.0,        8.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    # Pixel backward endcap disks to OT barrel L1 (CA 28)
    [ 16,  28,  False,  False,    1000,    11.6,   10000,    -57.5,    -30.0,   16.0,   -32.5,    -5.0,    0.85,    13.0,        5.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [ 17,  28,  False,  False,    1000,    11.6,   10000,    -80.0,    -40.0,   16.0,   -50.0,    -5.0,    0.85,    13.0,        5.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [ 18,  28,  False,  False,    1000,    11.6,   10000,    -95.0,    -55.0,   16.0,   -50.0,    -5.0,    0.85,    13.0,        8.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [ 19,  28,  False,  False,    1000,    11.8,   10000,   -110.0,    -70.0,   16.0,   -70.0,   -15.0,    0.85,    13.0,        8.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    [ 20,  28,  False,  False,    1000,       0,   10000,   -10000,    -80.0,   14.0,   -70.0,   -25.0,    0.85,    13.0,        8.0,        5.0,    -1.0,    0.003,     0.5,      0.0],
    # Pixel forward endcap connections
    [  4,   5,   True,  False,     300,       0,    14.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  4,   6,  False,   True,     522,       0,    14.0,      3.5,    10000,    9.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  5,   6,   True,  False,     300,       0,    13.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  5,   7,  False,   True,     522,       0,    13.0,      3.5,    10000,    9.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  6,   7,   True,  False,     250,       0,    13.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  6,   8,  False,   True,     522,       0,    13.0,      3.5,    10000,    9.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  7,   8,   True,  False,     250,       0,    13.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  7,   9,  False,   True,     522,       0,    13.0,      3.5,    10000,    8.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  8,   9,   True,  False,     250,       0,    13.0,      3.5,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  8,  10,  False,   True,     522,       0,    13.0,      3.5,    10000,    8.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  9,  10,   True,  False,     300,       0,    13.0,      4.0,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [  9,  11,  False,   True,     522,       0,    13.0,      4.0,    10000,    8.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 10,  11,   True,  False,     240,       0,    13.0,      3.5,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 10,  12,  False,   True,     650,    12.5,    16.5,     20.0,    10000,   10.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 11,  12,  False,  False,     300,       0,    16.5,      6.0,     21.0,    5.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 11,  13,  False,  False,     200,       0,     6.0,        0,      7.5,    3.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 11,  14,  False,  False,     220,       0,     4.6,        0,      7.5,    3.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 11,  15,  False,  False,     250,       0,     6.0,        0,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 12,  13,  False,  False,     250,       0,    22.5,      7.0,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 13,  14,  False,  False,     250,       0,    22.5,      7.0,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 14,  15,  False,  False,     250,       0,    22.5,      7.0,    10000,    3.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    # Pixel backward endcap connections
    [ 16,  17,   True,  False,     300,       0,    14.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 16,  18,  False,   True,     522,       0,    14.0,      3.5,    10000,    9.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 17,  18,   True,  False,     300,       0,    13.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 17,  19,  False,   True,     522,       0,    13.0,      3.5,    10000,    9.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 18,  19,   True,  False,     250,       0,    13.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 18,  20,  False,   True,     522,       0,    13.0,      3.5,    10000,    9.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 19,  20,   True,  False,     250,       0,    13.0,      3.5,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 19,  21,  False,   True,     522,       0,    13.0,      3.5,    10000,    8.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 20,  21,   True,  False,     250,       0,    13.0,      3.5,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 20,  22,  False,   True,     522,       0,    13.0,      3.5,    10000,    8.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 21,  22,   True,  False,     300,       0,    13.0,      4.0,    10000,    4.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 21,  23,  False,   True,     522,       0,    13.0,      4.0,    10000,    8.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 22,  23,   True,  False,     240,       0,    13.0,      3.5,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 22,  24,  False,   True,     650,    12.5,    16.5,     20.0,    10000,   10.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 23,  24,  False,  False,     300,       0,    16.5,      6.0,     21.0,    5.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 23,  25,  False,  False,     200,       0,     6.0,        0,      7.5,    3.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 23,  26,  False,  False,     220,       0,     4.6,        0,      7.5,    3.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 23,  27,  False,  False,     250,       0,     6.0,        0,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,    0.25,      0.0],
    [ 24,  25,  False,  False,     250,       0,    22.5,      7.0,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 25,  26,  False,  False,     250,       0,    22.5,      7.0,    10000,    4.0,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    [ 26,  27,  False,  False,     250,       0,    22.5,      7.0,    10000,    3.5,  -10000,   10000,    0.85,    13.0,       -1.0,       -1.0,    -1.0,    0.003,     0.4,      0.0],
    # OT barrel + endcap-disk connections; the odd-numbered 2S disk layers are enabled and connected below.
    # 2S rows: geometric purity cuts on (phiCut 1000, phiMid 0.05), stubSigma = geomKappa = 3.0, radii [55,105].
    # OT barrel to barrel connections (layers 28-33) - stubSigmaCut kappa-corrected significance
    [ 28,  29,  False,  False,    1100,   -1200,    1200,   -10000,    10000,  10000,   -50.0,    50.0,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,     0.5,      0.0],
    [ 29,  30,  False,  False,    1250,   -1200,    1200,   -10000,    10000,  10000,   -40.0,    40.0,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       2,      0.0],
    # caTheta on (30,31)/(31,32)/(32,33) follows TB2S sigma_z = 1.44 cm, against 0.043 cm for PS.
    [ 30,  31,  False,  False,    1250,  -10000,   10000,   -10000,    10000,  10000,   -30.0,    30.0,    0.85,    24.0,        5.0,        5.0,     0.1,   0.0365,       2,      0.0],
    [ 31,  32,  False,  False,    2000,  -10000,   10000,   -10000,    10000,  10000,   -30.0,    30.0,    0.85,    30.0,        5.0,        5.0,     0.1,   0.0886,       2,      0.0],
    [ 32,  33,  False,  False,    2000,  -10000,   10000,   -10000,    10000,  10000,   -25.0,    25.0,    0.85,    30.0,        5.0,        5.0,     0.1,   0.0902,     0.5,      0.0],
    # OT barrel (PS L1-L3) to the first z > 0 disk, PS layer (34)
    [ 28,  34,  False,  False,    1600,      40,     130,       20,       40,  10000,  -10000,   10000,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       3,   0.0002],
    [ 29,  34,  False,  False,    1700,      80,     130,       30,       60,  10000,  -10000,   10000,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       3,   0.0002],
    [ 30,  34,  False,  False,    2000,     100,     130,       50,       80,  10000,  -10000,   10000,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       3,   0.0002],
    # PS(barrel) -> 2S(disk) cross links
    [ 30,  35,  False,  False,    1000,      80,     108,       50,       80,  10000,  -10000,   10000,    0.85,    22.0,        3.0,        3.0,    0.05,  0.03712,       5,      0.0],
    [ 30,  45,  False,  False,    1000,    -108,     -80,       50,       80,  10000,  -10000,   10000,    0.85,    22.0,        3.0,        3.0,    0.05,  0.03712,       5,      0.0],
    # barrel-2S -> 2S(disk) at z > 0
    [ 31,  35,  False,  False,    1000,      90,     116,       60,      110,  10000,  -10000,   10000,    0.85,    34.0,        3.0,        3.0,    0.05,  0.06984,       5,      0.0],
    [ 32,  35,  False,  False,    1000,     101,     116,       80,      110,  10000,  -10000,   10000,    0.85,    46.0,        3.0,        3.0,    0.05,    0.096,       5,      0.0],
    # OT barrel (PS L1-L3) to the first z < 0 disk, PS layer (44)
    [ 28,  44,  False,  False,    2500,    -130,     -40,       20,       40,  10000,  -10000,   10000,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       3,   0.0002],
    [ 29,  44,  False,  False,    2500,    -130,     -80,       30,       60,  10000,  -10000,   10000,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       3,   0.0002],
    [ 30,  44,  False,  False,    2000,    -130,    -100,       50,       80,  10000,  -10000,   10000,    0.85,    13.0,        5.0,        5.0,     0.1,    0.005,       3,   0.0002],
    # barrel-2S -> 2S(disk) at z < 0
    [ 31,  45,  False,  False,    1000,    -116,     -90,       60,      110,  10000,  -10000,   10000,    0.85,    35.0,        3.0,        3.0,    0.05,  0.06984,       5,      0.0],
    [ 32,  45,  False,  False,    1000,    -116,     -99,       80,      110,  10000,  -10000,   10000,    0.85,    40.0,        3.0,        3.0,    0.05,  0.06984,       5,      0.0],
    # z > 0 disk-to-disk (layers 34-43): PS chain + PS->2S + 2S->2S
    [ 34,  36,  False,  False,    2500,      20,     115,       20,      115,   60.0,    15.0,    50.0,    0.85,    18.0,        6.5,        7.0,     0.1,     0.02,       3,   0.0002],
    [ 34,  37,  False,  False,    1000,      45,      72,       55,      105,  10000,  -10000,   10000,    0.85,    42.0,        3.0,        3.0,    0.05,   0.1312,       5,      0.0],  # PS->2S
    [ 36,  38,  False,  False,    2500,      20,     115,       20,      115,   60.0,    15.0,    50.0,    0.85,    18.0,        5.6,        7.0,     0.1,     0.02,       3,   0.0002],
    [ 36,  39,  False,  False,    1000,      45,      72,       55,      105,  10000,  -10000,   10000,    0.85,    47.0,        3.0,        3.0,    0.05,  0.06984,       5,      0.0],  # PS->2S
    [ 38,  40,  False,  False,    2500,      20,     115,       20,      115,   60.0,    15.0,    50.0,    0.85,    18.0,        5.4,        7.0,     0.1,     0.02,       3,   0.0002],
    [ 38,  41,  False,  False,    1000,      20,     115,       20,      115,   60.0,    15.0,    50.0,    0.85,    47.0,        3.0,        3.0,    0.05,  0.05096,       5,      0.0],  # PS->2S
    [ 40,  42,  False,  False,    2500,      20,     115,       20,      115,   60.0,    15.0,    50.0,    0.85,    16.0,        6.0,        7.0,     0.1,     0.02,    0.65,   0.0002],
    [ 40,  43,  False,  False,    1000,      20,     115,       20,      115,   60.0,    15.0,    50.0,    0.85,    48.0,        3.0,        3.0,    0.05,  0.05096,       3,   0.0002],  # PS->2S
    # 2S->2S chain at z > 0; caDCA is loose because the gate-all triplet DNN carries the purity here.
    [ 35,  37,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      47,        3.0,        3.0,    0.05,   0.1312,       5,      0.0],
    [ 37,  39,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      47,        3.0,        3.0,    0.05,   0.1312,       5,      0.0],
    [ 39,  41,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      45,        3.0,        3.0,    0.05,   0.1312,       5,      0.0],
    [ 41,  43,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      47,        3.0,        3.0,    0.05,   0.3392,       5,      0.0],
    # z < 0 disk-to-disk (layers 44-53): PS chain + PS->2S + 2S->2S
    [ 44,  46,  False,  False,    2500,      20,     115,       20,      115,   60.0,   -50.0,   -15.0,    0.85,    18.0,        6.5,        7.0,     0.1,     0.02,       3,   0.0002],
    [ 44,  47,  False,  False,    1000,      45,      72,       55,      105,  10000,  -10000,   10000,    0.85,    41.0,        3.0,        3.0,    0.05,  0.05096,       5,      0.0],  # PS->2S
    [ 46,  48,  False,  False,    2500,      20,     115,       20,      115,   60.0,   -50.0,   -15.0,    0.85,    18.0,        5.6,        7.0,     0.1,     0.02,       3,   0.0002],
    [ 46,  49,  False,  False,    1000,      45,      72,       55,      105,  10000,  -10000,   10000,    0.85,    47.0,        3.0,        3.0,    0.05,  0.06984,       5,      0.0],  # PS->2S
    [ 48,  50,  False,  False,    2500,      20,     115,       20,      115,   60.0,   -50.0,   -15.0,    0.85,    18.0,        5.4,        7.0,     0.1,     0.02,       3,   0.0002],
    [ 48,  51,  False,  False,    1000,      45,      72,       55,      105,  10000,  -10000,   10000,    0.85,    46.0,        3.0,        3.0,    0.05,  0.05096,       5,      0.0],  # PS->2S
    [ 50,  52,  False,  False,    2500,      20,     115,       20,      115,   60.0,   -50.0,   -15.0,    0.85,    15.0,        6.0,        7.0,     0.1,     0.02,    0.65,   0.0002],
    [ 50,  53,  False,  False,    1000,      45,      72,       55,      105,  10000,  -10000,   10000,    0.85,    47.0,        3.0,        3.0,    0.05,  0.06984,       3,      0.0],  # PS->2S
    # 2S->2S chain at z < 0, loose as the z > 0 chain above
    [ 45,  47,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      47,        3.0,        3.0,    0.05,    0.096,       5,      0.0],
    [ 47,  49,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      47,        3.0,        3.0,    0.05,   0.1312,       5,      0.0],
    [ 49,  51,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      45,        3.0,        3.0,    0.05,   0.1312,       5,      0.0],
    [ 51,  53,  False,  False,    1000,      55,     105,       55,      105,  10000,  -10000,   10000,    0.85,      47,        3.0,        3.0,    0.05,     0.18,       5,      0.0],
]

hltPhase2PixelTracksSoAWithStubs = cms.EDProducer('CAHitNtupletAlpakaPhase2OTStubs@alpaka',
    pixelRecHitSrc = cms.InputTag('hltPhase2PixelRecHitsStubsMerger'),

    # In-kernel classifiers: ON; their working points are BAKED in the weight headers (CATripletDNNWeights_*.h,
    # CATrackDNNWeights_*.h) by the retraining scripts, so no explicit threshold is set here (-1 = baked).
    useTripletDNN = cms.bool(True),
    useTrackDNN = cms.bool(True),

    # Generic cut scalars that differ from the topology defaults. Everything not listed here
    # (ptmin, dzdrFact, maxDYPred, the fishbone and duplicate-removal switches, the fit and DNN
    # flags, the covariance gate width) is left at the Phase2OTStubs default, which already is
    # this chain's working point.
    hardCurvCut = cms.double(0.02),
    minYsizeB1 = cms.int32(15),
    minYsizeB2 = cms.int32(14),
    maxDYsize12 = cms.int32(15),
    maxDYsize = cms.int32(20),
    # Max |phi residual| per connection during the chain extension [rad].
    maxPhiResid = cms.double(0.004),

    # Container sizes: maxNumberOf* are envelope fits of the per-event demand vs hit count, min/avg* the
    # floors and per-cell ratios, together ~15% of margin over PU200 occupancy. Values are for
    # useExactAllocations = False; the exact mode needs 0.1050 / 0.1096 for the same margin. An over-cap
    # event truncates without corrupting and is reported at endStream by the overflow sentinel.
    minNumberOfDoublets = cms.uint32(64768),  # quiet-event floor, inert at collision occupancy
    minNumberOfTuples = cms.uint32(65535),    # 65535, not 65536: hitContainerOffsets lands on the 256 KiB bin
    avgHitsPerTrack = cms.double(8.0),        # above the average, so stub-expanded (~13-hit) tracks fit
    avgCellsPerCell = cms.double(0.0984),
    avgTracksPerCell = cms.double(0.0997),
    # avgCellsPerHit is not set: hit->cell storage holds one entry per cell and is sized from the cell bound.
    maxNumberOfDoublets = cms.string("max(4.210056e-05*x*x+1.646969e+00*x,7.788906e+00*x)"),
    maxNumberOfTuples = cms.string("5.916732e-07*x*x"),

    # CA graph and cuts; caDCACuts / caThetaCuts are replaced on every pair by the per-pair overrides below.
    geometry = cms.PSet(
        maxDCurv    = cms.vdouble([l[3] for l in layers]),
        floorDCurv  = cms.vdouble([l[4] for l in layers]),
        fishboneCuts = cms.vdouble([l[5] for l in layers]),

        pairGraph     = cms.vuint32(sum([[lp[0], lp[1]] for lp in layerPairs], [])),
        startingPairs = cms.vuint32([i for i, lp in enumerate(layerPairs) if lp[2]]),
        skipsLayers   = cms.vuint32([int(lp[3]) for lp in layerPairs]),
        phiCuts  = cms.vint32( [lp[ 4] for lp in layerPairs]),
        minInner = cms.vdouble([lp[ 5] for lp in layerPairs]),
        maxInner = cms.vdouble([lp[ 6] for lp in layerPairs]),
        minOuter = cms.vdouble([lp[ 7] for lp in layerPairs]),
        maxOuter = cms.vdouble([lp[ 8] for lp in layerPairs]),
        maxDR    = cms.vdouble([lp[ 9] for lp in layerPairs]),
        minDZ    = cms.vdouble([lp[10] for lp in layerPairs]),
        maxDZ    = cms.vdouble([lp[11] for lp in layerPairs]),
        ptCuts   = cms.vdouble([lp[12] for lp in layerPairs]),

        # OT-stub extension: per-layer-pair overrides and stub-only cuts
        cellZ0CutPerPair         = cms.vdouble([lp[13] for lp in layerPairs]),
        maxStubCurvSigma         = cms.vdouble([lp[14] for lp in layerPairs]),
        maxStubGeomCurvSigma     = cms.vdouble([lp[15] for lp in layerPairs]),
        maxStubInnerDoubletDCurv = cms.vdouble([lp[16] for lp in layerPairs]),
        caThetaCutsPerPair       = cms.vdouble([lp[17] for lp in layerPairs]),
        caDCACutsPerPair         = cms.vdouble([lp[18] for lp in layerPairs]),
        floorDCACuts             = cms.vdouble([lp[19] for lp in layerPairs]),
    ),

    # autoselect the alpaka backend
    alpaka = cms.untracked.PSet(backend = cms.untracked.string(''))
)

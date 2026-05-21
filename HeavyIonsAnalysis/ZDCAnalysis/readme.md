# Map (Small system run, July 2025)
ZDC and FSC channel map and by default if they are saved in the 3 trees in the forest.

| idet | zside | section | channel | Detector | ZDCrechit | ZDCdigi | FSCdigi |
| -----: | -----: | -----: | -----: | :----- | :-----: | :-----: | :-----: |
|  0 | -1 |  1 |  1 | ZDCm EM 1 | :heavy_check_mark: | :heavy_check_mark: | |
|  1 | -1 |  1 |  2 | ZDCm EM 2 | :heavy_check_mark: | :heavy_check_mark: | |
|  2 | -1 |  1 |  3 | ZDCm EM 3 | :heavy_check_mark: | :heavy_check_mark: | |
|  3 | -1 |  1 |  4 | ZDCm EM 4 | :heavy_check_mark: | :heavy_check_mark: | |
|  4 | -1 |  1 |  5 | ZDCm EM 5 | :heavy_check_mark: | :heavy_check_mark: | |
|  5 | -1 |  1 |  6 | Dummy | | | :heavy_check_mark: |
|  6 | -1 |  1 |  7 | FSC2 Up | | | :heavy_check_mark: |
|  7 | -1 |  1 |  8 | FSC2 Down | | | :heavy_check_mark: |
|  8 | -1 |  1 |  9 | FSC3 BottLeft | | | :heavy_check_mark: |
|  9 | -1 |  1 | 10 | FSC3 BottRight | | | :heavy_check_mark: |
| 10 | -1 |  1 | 11 | FSC3 Topleft | | | :heavy_check_mark: |
| 11 | -1 |  1 | 12 | FSC3 TopRight | | | :heavy_check_mark: |
| 12 | -1 |  2 |  1 | ZDCm HAD 1 | :heavy_check_mark: | :heavy_check_mark: | |
| 13 | -1 |  2 |  2 | ZDCm HAD 2 | :heavy_check_mark: | :heavy_check_mark: | |
| 14 | -1 |  2 |  3 | ZDCm HAD 3 | :heavy_check_mark: | :heavy_check_mark: | |
| 15 | -1 |  2 |  4 | ZDCm HAD 4 | :heavy_check_mark: | :heavy_check_mark: | |
| 16 |  1 |  1 |  1 | ZDCp EM 1 | :heavy_check_mark: | :heavy_check_mark: | |
| 17 |  1 |  1 |  2 | ZDCp EM 2 | :heavy_check_mark: | :heavy_check_mark: | |
| 18 |  1 |  1 |  3 | ZDCp EM 3 | :heavy_check_mark: | :heavy_check_mark: | |
| 19 |  1 |  1 |  4 | ZDCp EM 4 | :heavy_check_mark: | :heavy_check_mark: | |
| 20 |  1 |  1 |  5 | ZDCp EM 5 | :heavy_check_mark: | :heavy_check_mark: | |
| 21 |  1 |  1 |  6 | Dummy | | | :heavy_check_mark: |
| 22 |  1 |  1 |  7 | Dummy | | | :heavy_check_mark: |
| 23 |  1 |  1 |  8 | Dummy | | | :heavy_check_mark: |
| 24 |  1 |  2 |  1 | ZDCp HAD 1 | :heavy_check_mark: | :heavy_check_mark: | |
| 25 |  1 |  2 |  2 | ZDCp HAD 2 | :heavy_check_mark: | :heavy_check_mark: | |
| 26 |  1 |  2 |  3 | ZDCp HAD 3 | :heavy_check_mark: | :heavy_check_mark: | |
| 27 |  1 |  2 |  4 | ZDCp HAD 4 | :heavy_check_mark: | :heavy_check_mark: | |
| 28-43 | -1 |  4 |  1-16 | ZDCm RPD 1-16 | | :heavy_check_mark: | |
| 44-59 |  1 |  4 |  1-16 | ZDCp RPD 1-16 | | :heavy_check_mark: | |

---

# ZDC analyzer
## Minimum usage
Energy sum is `(float) sumPlus` and `(float) sumMinus` in `zdcanalyzer/zdcrechit`.
```
   (HiForestMiniAOD.root)
   ./
    └── (TDirectoryFile) => zdcanalyzer
        └── (TTree) => zdcrechit (1)
```

## Indices and dimensions
- By default, the 18 channels of ZDC are saved in rechit tree, while additional 32 RPD channels are saved in digi tree.
```
   (HiForestMiniAOD.root)
   ./
    └── (TDirectoryFile) => zdcanalyzer
        ├── (TTree) => zdcrechit (1)
***********************************************************************
*    Row   * Instance *         n *     zside *   section *   channel *
***********************************************************************
*        0 *        0 *        18 *        -1 *         1 *         1 *
*        0 *        1 *        18 *        -1 *         1 *         2 *
*        0 *        2 *        18 *        -1 *         1 *         3 *
*        0 *        3 *        18 *        -1 *         1 *         4 *
*        0 *        4 *        18 *        -1 *         1 *         5 *
*        0 *        5 *        18 *        -1 *         2 *         1 *
*        0 *        6 *        18 *        -1 *         2 *         2 *
*        0 *        7 *        18 *        -1 *         2 *         3 *
*        0 *        8 *        18 *        -1 *         2 *         4 *
*        0 *        9 *        18 *         1 *         1 *         1 *
*        0 *       10 *        18 *         1 *         1 *         2 *
*        0 *       11 *        18 *         1 *         1 *         3 *
*        0 *       12 *        18 *         1 *         1 *         4 *
*        0 *       13 *        18 *         1 *         1 *         5 *
*        0 *       14 *        18 *         1 *         2 *         1 *
*        0 *       15 *        18 *         1 *         2 *         2 *
*        0 *       16 *        18 *         1 *         2 *         3 *
*        0 *       17 *        18 *         1 *         2 *         4 *
***********************************************************************

        └── (TTree) => zdcdigi (1)
***********************************************************************
*    Row   * Instance *         n *     zside *   section *   channel *
***********************************************************************
*        0 *        0 *        50 *        -1 *         1 *         1 *
*        0 *        1 *        50 *        -1 *         1 *         2 *
*        0 *        2 *        50 *        -1 *         1 *         3 *
*        0 *        3 *        50 *        -1 *         1 *         4 *
*        0 *        4 *        50 *        -1 *         1 *         5 *
*        0 *        5 *        50 *        -1 *         2 *         1 *
*        0 *        6 *        50 *        -1 *         2 *         2 *
*        0 *        7 *        50 *        -1 *         2 *         3 *
*        0 *        8 *        50 *        -1 *         2 *         4 *
*        0 *        9 *        50 *         1 *         1 *         1 *
*        0 *       10 *        50 *         1 *         1 *         2 *
*        0 *       11 *        50 *         1 *         1 *         3 *
*        0 *       12 *        50 *         1 *         1 *         4 *
*        0 *       13 *        50 *         1 *         1 *         5 *
*        0 *       14 *        50 *         1 *         2 *         1 *
*        0 *       15 *        50 *         1 *         2 *         2 *
*        0 *       16 *        50 *         1 *         2 *         3 *
*        0 *       17 *        50 *         1 *         2 *         4 *
*        0 *       18 *        50 *        -1 *         4 *         1 *
*        0 *       19 *        50 *        -1 *         4 *         2 *
*        0 *       20 *        50 *        -1 *         4 *         3 *
*        0 *       21 *        50 *        -1 *         4 *         4 *
*        0 *       22 *        50 *        -1 *         4 *         5 *
*        0 *       23 *        50 *        -1 *         4 *         6 *
*        0 *       24 *        50 *        -1 *         4 *         7 *
*        0 *       25 *        50 *        -1 *         4 *         8 *
*        0 *       26 *        50 *        -1 *         4 *         9 *
*        0 *       27 *        50 *        -1 *         4 *        10 *
*        0 *       28 *        50 *        -1 *         4 *        11 *
*        0 *       29 *        50 *        -1 *         4 *        12 *
*        0 *       30 *        50 *        -1 *         4 *        13 *
*        0 *       31 *        50 *        -1 *         4 *        14 *
*        0 *       32 *        50 *        -1 *         4 *        15 *
*        0 *       33 *        50 *        -1 *         4 *        16 *
*        0 *       34 *        50 *         1 *         4 *         1 *
*        0 *       35 *        50 *         1 *         4 *         2 *
*        0 *       36 *        50 *         1 *         4 *         3 *
*        0 *       37 *        50 *         1 *         4 *         4 *
*        0 *       38 *        50 *         1 *         4 *         5 *
*        0 *       39 *        50 *         1 *         4 *         6 *
*        0 *       40 *        50 *         1 *         4 *         7 *
*        0 *       41 *        50 *         1 *         4 *         8 *
*        0 *       42 *        50 *         1 *         4 *         9 *
*        0 *       43 *        50 *         1 *         4 *        10 *
*        0 *       44 *        50 *         1 *         4 *        11 *
*        0 *       45 *        50 *         1 *         4 *        12 *
*        0 *       46 *        50 *         1 *         4 *        13 *
*        0 *       47 *        50 *         1 *         4 *        14 *
*        0 *       48 *        50 *         1 *         4 *        15 *
*        0 *       49 *        50 *         1 *         4 *        16 *
***********************************************************************
```
- The safest length of the arrays is 60 = (9 ZDC + 16 RPD) &times; 2 sides + 4 dump (3 at plus + 1 at minus) + 6 FSC, no matter if RPD is skipped, e.g.
```
#define MAXMOD 60 
int zside[MAXMOD];
```

## Options and parameters
### ZDC analyzer
- Hard coded RPD `doHardcodedRPD`: Geometry updated for the RPD are not part of 14_1_X and the GT used for 2024. Always do hard coded RPD for 2024.
- Remove digi tree: `process.zdcanalyzer.doZdcDigis = cms.bool(False)`
- Save RPD rechit: `process.zdcanalyzer.skipRpdRecHits = cms.bool(False)`
- Remove RPD digi: `process.zdcanalyzer.skipRpdDigis = cms.bool(True)`
- Add Aux
```
process.zdcanalyzer.doAuxZdcRecHits = cms.bool(True)
process.zdcanalyzer.AuxZDCRecHitSource = cms.InputTag('your zdc rechit label')
```

# ZDC rechit producer
## Options and parameters
| | correctionMethodHAD | correctionMethodEM | ootpuRatioHAD | ootpuRatioEM | ootpuFracHAD | ootpuFracEM |
| :-- | :--: | :--: | :--: | :--: | :--: | :--: |
| Default in `zdcrecoRun3_cfi` (Template fit) | 1 | 1 | 3.0 | 3.0 | 0.4 | 0.4 |
| Trigger (`Ts2-0.4*Ts1`) | 0 | 0 | -1 | -1 | 97.0/256.0 | 97.0/256.0 |
| 2023 offline (`Ts2-Ts1`), default in forest | 0 | 0 | -1 | -1 | 1.0 | 1.0 |

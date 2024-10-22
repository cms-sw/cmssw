# simple test for the BunchCrossingFilter

# colliding bunches in run 317435, fill 6759, scheme 25ns_2556b_2544_2215_2332_144bpi_20injV2:
#     81-128,   136-183
#    215-262,   270-317,   325-372,
#    404-451,   459-506,   514-561,
#    593-640,   648-695,   703-750,
#    786-833,   841-888,
#    920-967,   975-1022, 1030-1077,
#   1109-1156, 1164-1211, 1219-1266,
#   1298-1345, 1353-1400, 1408-1455,
#   1487-1534, 1542-1589, 1597-1644,
#   1680-1727, 1735-1782,
#   1814-1861, 1869-1916, 1924-1971,
#   2003-2050, 2058-2105, 2113-2160,
#   2192-2239, 2247-2294, 2302-2349,
#   2381-2428, 2436-2483, 2491-2538,
#   2574-2621, 2629-2676,
#   2708-2755, 2763-2810, 2818-2865,
#   2897-2944, 2952-2999, 3007-3054,
#   3086-3133, 3141-3188, 3196-3243,
#   3275-3322, 3330-3377, 3385-3432
# (see https://lpc.web.cern.ch/fillingSchemes/2018/25ns_2556b_2544_2215_2332_144bpi_20injV2.csv)

from builtins import range
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.pfnInPath import *

process = cms.Process('TEST')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.load('FWCore.MessageService.MessageLogger_cfi')

# input data
process.source = cms.Source('PoolSource',
    fileNames = cms.untracked.pfnInPaths('FWCore/Modules/data/rawData_empty_CMSSW_10_2_0.root')
)

from FWCore.Modules.bunchCrossingFilter_cfi import bunchCrossingFilter as _bunchCrossingFilter

# empty input, do not select any bunch crossings
process.selectNone = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(),
)

# full range of possible bunch crossings [1,3564]
process.selectAll = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(list(range(1,3565)))
)

# select bx 536
process.selectSingle = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(536)
)

# select the whole train 514-561
process.selectTrain = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(list(range(514,562)))
)

# inverted to veto (non-colliding) bx 1
process.selectEmpty = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(1)
)

process.SelectNone        = cms.Path( process.selectNone )
process.SelectAll         = cms.Path( process.selectAll )
process.SelectSingle      = cms.Path( process.selectSingle )
process.SelectTrain       = cms.Path( process.selectTrain )
process.VetoEmpty         = cms.Path( ~ process.selectEmpty )
process.VetoSingle        = cms.Path( ~ process.selectSingle )
process.VetoTrain         = cms.Path( ~ process.selectTrain )
process.SelectTrainButOne = cms.Path( process.selectTrain * ~ process.selectSingle )

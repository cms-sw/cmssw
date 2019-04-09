import FWCore.ParameterSet.Config as cms
#from test.ParameterSet.pfnInPath import *

from FWCore.Modules.bunchCrossingFilter_cfi import bunchCrossingFilter as _bunchCrossingFilter

# empty input, do not select any bunch crossings
process.selectNone = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(),
)

test = _bunchCrossingFilter.clone(
    bunches = cms.vuint32(3),
)

# full range of possible bunch crossings [1,3564]
process.selectAll = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(range(757,759))
)

# select bx 536
process.selectSingle = _bunchCrossingFilter.clone(
   bunches = cms.vuint32(755)
)

# select the whole train 514-561
process.selectTrain = _bunchCrossingFilter.clone(
    bunches = cms.vuint32(range(1646,1651)) 
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
process.SelectTrainButOne = cms.Path( process.selectTrain *  process.selectSingle * process.selectAll)


BunchCrossingSequence = cms.Sequence(
    test
)


BunchCrossingSeqtest = cms.Sequence(    
    selectAll
    *SelectSingle
    *selectAll
)





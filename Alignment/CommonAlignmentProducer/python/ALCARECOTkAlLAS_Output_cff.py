import FWCore.ParameterSet.Config as cms

# output block for LAS AlCaReco
OutALCARECOTkAlLAS_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlLAS')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlLAST0Producer_*_*', 
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

import copy
OutALCARECOTkAlLAS = copy.deepcopy(OutALCARECOTkAlLAS_noDrop)
OutALCARECOTkAlLAS.outputCommands.insert(0, "drop *")

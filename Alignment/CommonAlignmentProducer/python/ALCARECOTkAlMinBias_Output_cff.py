import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MinBias events
OutALCARECOTkAlMinBias_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlMinBias')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlMinBias_*_*', 
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlMinBias = copy.deepcopy(OutALCARECOTkAlMinBias_noDrop)
OutALCARECOTkAlMinBias.outputCommands.insert(0, "drop *")

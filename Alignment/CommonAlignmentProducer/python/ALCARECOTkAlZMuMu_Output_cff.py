import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events
OutALCARECOTkAlZMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlZMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlZMuMu_*_*', 
        'keep *_MEtoEDMConverter_*_*')
)

import copy
OutALCARECOTkAlZMuMu = copy.deepcopy(OutALCARECOTkAlZMuMu_noDrop)
OutALCARECOTkAlZMuMu.outputCommands.insert(0, "drop *")

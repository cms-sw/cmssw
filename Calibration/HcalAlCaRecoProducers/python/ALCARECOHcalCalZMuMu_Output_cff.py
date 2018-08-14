import FWCore.ParameterSet.Config as cms

# AlCaReco for HO using ZMuMu events
OutALCARECOHcalCalZMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOHcalCalZMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOHcalCalZMuMu_*_*', 
        'keep HcalNoiseSummary_hcalnoise_*_*',
        'keep *_horeco_*_*')
)

import copy
OutALCARECOHcalCalZMuMu=copy.deepcopy(OutALCARECOHcalCalZMuMu_noDrop)
OutALCARECOHcalCalZMuMu.outputCommands.insert(0, "drop *")

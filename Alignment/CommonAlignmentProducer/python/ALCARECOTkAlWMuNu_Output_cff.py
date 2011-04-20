import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using WMuNu events
OutALCARECOTkAlWMuNu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlWMuNu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlWMuNu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

import copy
OutALCARECOTkAlWMuNu = copy.deepcopy(OutALCARECOTkAlWMuNu_noDrop)
OutALCARECOTkAlWMuNu.outputCommands.insert(0, "drop *")

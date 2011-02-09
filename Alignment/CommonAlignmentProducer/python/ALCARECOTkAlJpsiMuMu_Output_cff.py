import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using JpsiMuMu events
OutALCARECOTkAlJpsiMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlJpsiMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlJpsiMuMu_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

import copy
OutALCARECOTkAlJpsiMuMu = copy.deepcopy(OutALCARECOTkAlJpsiMuMu_noDrop)
OutALCARECOTkAlJpsiMuMu.outputCommands.insert(0, "drop *")

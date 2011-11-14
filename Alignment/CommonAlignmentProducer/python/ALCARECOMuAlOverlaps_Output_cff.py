import FWCore.ParameterSet.Config as cms

# AlCaReco for muon based alignment using tracks in the CSC overlap regions
OutALCARECOMuAlOverlaps_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOMuAlOverlaps')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOMuAlOverlaps_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

import copy
OutALCARECOMuAlOverlaps = copy.deepcopy(OutALCARECOMuAlOverlaps_noDrop)
OutALCARECOMuAlOverlaps.outputCommands.insert(0, "drop *")

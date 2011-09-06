import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MinBias events
OutALCARECOTkAlMinBiasHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlMinBiasHI')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlMinBiasHI_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_offlineBeamSpot_*_*')
)

import copy
OutALCARECOTkAlMinBiasHI = copy.deepcopy(OutALCARECOTkAlMinBiasHI_noDrop)
OutALCARECOTkAlMinBiasHI.outputCommands.insert(0, "drop *")

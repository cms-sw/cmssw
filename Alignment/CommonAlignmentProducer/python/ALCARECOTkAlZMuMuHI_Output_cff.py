import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Z->mumu events in heavy ion (PbPb) data
OutALCARECOTkAlZMuMuHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlZMuMuHI')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlZMuMuHI_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_hiSelectedVertex_*_*')
)

import copy
OutALCARECOTkAlZMuMuHI = copy.deepcopy(OutALCARECOTkAlZMuMuHI_noDrop)
OutALCARECOTkAlZMuMuHI.outputCommands.insert(0, "drop *")

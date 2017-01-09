import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Z->mumu events in heavy ion (PA) data 
OutALCARECOTkAlZMuMuPA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlZMuMuPA')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlZMuMuPA_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

import copy
OutALCARECOTkAlZMuMuPA = copy.deepcopy(OutALCARECOTkAlZMuMuPA_noDrop)
OutALCARECOTkAlZMuMuPA.outputCommands.insert(0, "drop *")

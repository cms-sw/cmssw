import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion (PA) data 
OutALCARECOTkAlUpsilonMuMuPA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlUpsilonMuMuPA')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOTkAlUpsilonMuMuPA_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

import copy
OutALCARECOTkAlUpsilonMuMuPA = copy.deepcopy(OutALCARECOTkAlUpsilonMuMuPA_noDrop)
OutALCARECOTkAlUpsilonMuMuPA.outputCommands.insert(0, "drop *")

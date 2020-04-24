import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion (PbPb) data
OutALCARECOTkAlUpsilonMuMuHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlUpsilonMuMuHI')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOTkAlUpsilonMuMuHI_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_hiSelectedVertex_*_*')
)

import copy
OutALCARECOTkAlUpsilonMuMuHI = copy.deepcopy(OutALCARECOTkAlUpsilonMuMuHI_noDrop)
OutALCARECOTkAlUpsilonMuMuHI.outputCommands.insert(0, "drop *")

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using UpsilonMuMu events
OutALCARECOTkAlUpsilonMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlUpsilonMuMu')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOTkAlUpsilonMuMu_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

import copy
OutALCARECOTkAlUpsilonMuMu = copy.deepcopy(OutALCARECOTkAlUpsilonMuMu_noDrop)
OutALCARECOTkAlUpsilonMuMu.outputCommands.insert(0, "drop *")

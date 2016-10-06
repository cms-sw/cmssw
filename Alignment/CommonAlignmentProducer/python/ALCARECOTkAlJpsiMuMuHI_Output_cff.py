import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Jpsi->mumu events in heavy ion (PbPb) data
OutALCARECOTkAlJpsiMuMuHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlJpsiMuMuHI')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_ALCARECOTkAlJpsiMuMuHI_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_hiSelectedVertex_*_*')
)

import copy
OutALCARECOTkAlJpsiMuMuHI = copy.deepcopy(OutALCARECOTkAlJpsiMuMuHI_noDrop)
OutALCARECOTkAlJpsiMuMuHI.outputCommands.insert(0, "drop *")

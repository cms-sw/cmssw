import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MuonIsolated events for heavy ion (PbPb) data
OutALCARECOTkAlMuonIsolatedHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlMuonIsolatedHI')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOTkAlMuonIsolatedHI_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_hiSelectedVertex_*_*')
)

import copy
OutALCARECOTkAlMuonIsolatedHI = copy.deepcopy(OutALCARECOTkAlMuonIsolatedHI_noDrop)
OutALCARECOTkAlMuonIsolatedHI.outputCommands.insert(0, "drop *")

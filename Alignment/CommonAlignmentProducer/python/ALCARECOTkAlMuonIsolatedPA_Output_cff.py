import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MuonIsolated events
OutALCARECOTkAlMuonIsolatedPA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlMuonIsolatedPA')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOTkAlMuonIsolatedPA_*_*', 
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

import copy
OutALCARECOTkAlMuonIsolatedPA = copy.deepcopy(OutALCARECOTkAlMuonIsolatedPA_noDrop)
OutALCARECOTkAlMuonIsolatedPA.outputCommands.insert(0, "drop *")

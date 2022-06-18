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

# in Run3, SCAL digis replaced by onlineMetaDataDigis
_run3_common_removedCommands = OutALCARECOTkAlMuonIsolatedHI.outputCommands
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlMuonIsolatedHI, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

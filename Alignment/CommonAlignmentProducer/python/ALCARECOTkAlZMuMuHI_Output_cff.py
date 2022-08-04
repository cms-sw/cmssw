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

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlZMuMuHI_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlZMuMuHI_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

OutALCARECOTkAlZMuMuHI = OutALCARECOTkAlZMuMuHI_noDrop.clone()
OutALCARECOTkAlZMuMuHI.outputCommands.insert(0, "drop *")

import FWCore.ParameterSet.Config as cms

# AlCaReco for track based calibration using Cosmics events
OutALCARECOSiStripCalCosmics_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOSiStripCalCosmics')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep *_ALCARECOSiStripCalCosmics_*_*', 
        'keep *_siStripClusters_*_*', 
        'keep *_siPixelClusters_*_*',
        'keep DetIds_siStripDigis_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep LumiScalerss_scalersRawToDigi_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_TriggerResults_*_*')
    )

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOSiStripCalCosmics_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep LumiScalerss_scalersRawToDigi_*_*')
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*',
                              'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOSiStripCalCosmics_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

OutALCARECOSiStripCalCosmics=OutALCARECOSiStripCalCosmics_noDrop.clone()
OutALCARECOSiStripCalCosmics.outputCommands.insert(0,"drop *")

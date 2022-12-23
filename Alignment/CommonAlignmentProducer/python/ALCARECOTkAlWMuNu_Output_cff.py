import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using WMuNu events
OutALCARECOTkAlWMuNu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlWMuNu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlWMuNu_*_*',
        'keep recoTrackExtras_ALCARECOTkAlWMuNu_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlWMuNu_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlWMuNu_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlWMuNu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlWMuNu_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*',
                              'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlWMuNu_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlWMuNu_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlWMuNu_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlWMuNu_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlWMuNu_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlWMuNu = OutALCARECOTkAlWMuNu_noDrop.clone()
OutALCARECOTkAlWMuNu.outputCommands.insert(0, "drop *")


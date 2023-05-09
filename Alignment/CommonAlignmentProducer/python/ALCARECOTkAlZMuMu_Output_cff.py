import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events
OutALCARECOTkAlZMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlZMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlZMuMu_*_*',
        'keep recoTrackExtras_ALCARECOTkAlZMuMu_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlZMuMu_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlZMuMu_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlZMuMu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlZMuMu_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlZMuMu_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlZMuMu_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlZMuMu_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlZMuMu_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlZMuMu_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlZMuMu = OutALCARECOTkAlZMuMu_noDrop.clone()
OutALCARECOTkAlZMuMu.outputCommands.insert(0, "drop *")

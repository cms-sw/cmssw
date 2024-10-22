import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using BeamHalo events
OutALCARECOTkAlBeamHalo_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlBeamHalo')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlBeamHalo_*_*',
        'keep recoTrackExtras_ALCARECOTkAlBeamHalo_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlBeamHalo_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlBeamHalo_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlBeamHalo_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlBeamHalo_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*',
                              'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlBeamHalo_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlBeamHalo_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlBeamHalo_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlBeamHalo_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlBeamHalo_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlBeamHalo = copy.deepcopy(OutALCARECOTkAlBeamHalo_noDrop)
OutALCARECOTkAlBeamHalo.outputCommands.insert(0, "drop *")

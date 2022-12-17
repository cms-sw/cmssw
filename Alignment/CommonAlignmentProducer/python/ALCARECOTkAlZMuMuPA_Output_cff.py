import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Z->mumu events in heavy ion (PA) data 
OutALCARECOTkAlZMuMuPA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlZMuMuPA')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlZMuMuPA_*_*',
        'keep recoTrackExtras_ALCARECOTkAlZMuMuPA_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlZMuMuPA_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlZMuMuPA_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlZMuMuPA_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlZMuMuPA_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*',
                              'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlZMuMuPA_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlZMuMuPA_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlZMuMuPA_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlZMuMuPA_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlZMuMuPA_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlZMuMuPA = copy.deepcopy(OutALCARECOTkAlZMuMuPA_noDrop)
OutALCARECOTkAlZMuMuPA.outputCommands.insert(0, "drop *")

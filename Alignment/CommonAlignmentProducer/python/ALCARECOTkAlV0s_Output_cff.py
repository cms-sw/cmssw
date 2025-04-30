import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MinBias events
OutALCARECOTkAlV0s_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlK0s',
                                   'pathALCARECOTkAlLambdas')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlKShortTracks_*_*',
        'keep recoTrackExtras_ALCARECOTkAlKShortTracks_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlKShortTracks_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlKShortTracks_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlKShortTracks_*_*',
        'keep recoTracks_ALCARECOTkAlLambdaTracks_*_*',
        'keep recoTrackExtras_ALCARECOTkAlLambdaTracks_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlLambdaTracks_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlLambdaTracks_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlLambdaTracks_*_*',
        'keep *_generalV0Candidates_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_offlineBeamSpot_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlV0s_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*',
                              'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlV0s_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlV0s_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlKShortTracks_*_*')
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlLambdaTracks_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlKShortTracks_*_*',
                                'keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlLambdaTracks_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlV0s_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlV0s = OutALCARECOTkAlV0s_noDrop.clone()
OutALCARECOTkAlV0s.outputCommands.insert(0, "drop *")
-- dummy change --

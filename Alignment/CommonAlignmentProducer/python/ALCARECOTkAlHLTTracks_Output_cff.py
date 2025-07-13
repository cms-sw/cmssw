import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MinBias events
OutALCARECOTkAlHLTTracks_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlHLTTracks')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlHLTTracks_*_*',
        'keep recoTrackExtras_ALCARECOTkAlHLTTracks_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlHLTTracks_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlHLTTracks_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlHLTTracks_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_hltPixelVertices_*_*',
        'keep *_hltVerticesPFFilter_*_*',
        'keep *_hltOnlineBeamSpot_*_*',
        'keep DCSRecord_onlineMetaDataDigis_*_*'
    )
)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlHLTTracks_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlHLTTracks_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlHLTTracks_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlHLTTracks_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlHLTTracks = OutALCARECOTkAlHLTTracks_noDrop.clone()
OutALCARECOTkAlHLTTracks.outputCommands.insert(0, "drop *")

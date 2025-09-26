import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events
OutALCARECOTkAlHLTTracksZMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlHLTTracksZMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep recoTrackExtras_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlHLTTracksZMuMu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_hltPixelVertices_*_*',
        'keep recoTracks_ALCARECOTkAlHLTPixelZMuMuVertexTracks_*_*',
        'keep recoTrackExtras_ALCARECOTkAlHLTPixelZMuMuVertexTracks_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlHLTPixelZMuMuVertexTracks_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlHLTPixelZMuMuVertexTracks_*_*',
	'keep *_hltVerticesPFFilter_*_*',
        'keep *_hltOnlineBeamSpot_*_*',
        'keep DCSRecord_onlineMetaDataDigis_*_*'
    )
)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlHLTTracksZMuMu_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlHLTTracksZMuMu_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlHLTTracksZMuMu_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlHLTTracksZMuMu_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlHLTTracksZMuMu = OutALCARECOTkAlHLTTracksZMuMu_noDrop.clone()
OutALCARECOTkAlHLTTracksZMuMu.outputCommands.insert(0, "drop *")


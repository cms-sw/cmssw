import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using ZMuMu events (including the tracks from the PV)
OutALCARECOTkAlDiMuonAndVertex_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlDiMuonAndVertex')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlDiMuon_*_*',
        'keep recoTrackExtras_ALCARECOTkAlDiMuon_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlDiMuon_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlDiMuon_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlDiMuon_*_*',
        'keep recoTracks_ALCARECOTkAlDiMuonVertexTracks_*_*',
        'keep recoTrackExtras_ALCARECOTkAlDiMuonVertexTracks_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlDiMuonVertexTracks_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlDiMuonVertexTracks_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlDiMuonVertexTracks_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
        'keep *_offlinePrimaryVertices_*_*')
)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlDiMuonAndVertex_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlDiMuon_*_*')
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlDiMuonVertexTracks_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlDiMuon_*_*',
                                'keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlDiMuonVertexTracks_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlDiMuonAndVertex_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlDiMuonAndVertex = OutALCARECOTkAlDiMuonAndVertex_noDrop.clone()
OutALCARECOTkAlDiMuonAndVertex.outputCommands.insert(0, "drop *")

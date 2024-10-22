import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using MuonIsolated events
OutALCARECOTkAlMuonIsolatedPA_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlMuonIsolatedPA')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep recoTracks_ALCARECOTkAlMuonIsolatedPA_*_*',
        'keep recoTrackExtras_ALCARECOTkAlMuonIsolatedPA_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlMuonIsolatedPA_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlMuonIsolatedPA_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlMuonIsolatedPA_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

import copy
_run3_common_removedCommands = OutALCARECOTkAlMuonIsolatedPA_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*',
                              'keep OnlineLuminosityRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlMuonIsolatedPA_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlMuonIsolatedPA_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlMuonIsolatedPA_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlMuonIsolatedPA_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlMuonIsolatedPA_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlMuonIsolatedPA = copy.deepcopy(OutALCARECOTkAlMuonIsolatedPA_noDrop)
OutALCARECOTkAlMuonIsolatedPA.outputCommands.insert(0, "drop *")

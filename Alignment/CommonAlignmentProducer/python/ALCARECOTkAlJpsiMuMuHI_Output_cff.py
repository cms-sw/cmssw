import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Jpsi->mumu events in heavy ion (PbPb) data
OutALCARECOTkAlJpsiMuMuHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlJpsiMuMuHI')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlJpsiMuMuHI_*_*',
        'keep recoTrackExtras_ALCARECOTkAlJpsiMuMuHI_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlJpsiMuMuHI_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMuHI_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMuHI_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_hiSelectedVertex_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlJpsiMuMuHI_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlJpsiMuMuHI_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlJpsiMuMuHI_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMuHI_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlJpsiMuMuHI_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlJpsiMuMuHI_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlJpsiMuMuHI = OutALCARECOTkAlJpsiMuMuHI_noDrop.clone()
OutALCARECOTkAlJpsiMuMuHI.outputCommands.insert(0, "drop *")

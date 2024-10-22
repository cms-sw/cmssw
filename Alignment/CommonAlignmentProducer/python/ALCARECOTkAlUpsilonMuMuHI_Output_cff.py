import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using Upsilon->mumu events in heavy ion (PbPb) data
OutALCARECOTkAlUpsilonMuMuHI_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlUpsilonMuMuHI')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep recoTracks_ALCARECOTkAlUpsilonMuMuHI_*_*',
        'keep recoTrackExtras_ALCARECOTkAlUpsilonMuMuHI_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlUpsilonMuMuHI_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlUpsilonMuMuHI_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlUpsilonMuMuHI_*_*',        
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_hiSelectedVertex_*_*')
)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlUpsilonMuMuHI_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlUpsilonMuMuHI_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlUpsilonMuMuHI_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlUpsilonMuMuHI_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlUpsilonMuMuHI_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlUpsilonMuMuHI_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlUpsilonMuMuHI = OutALCARECOTkAlUpsilonMuMuHI_noDrop.clone()
OutALCARECOTkAlUpsilonMuMuHI.outputCommands.insert(0, "drop *")

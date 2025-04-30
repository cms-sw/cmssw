import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using JpsiMuMu events
OutALCARECOTkAlJpsiMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlJpsiMuMu')
    ),
    outputCommands = cms.untracked.vstring(
        'keep recoTracks_ALCARECOTkAlJpsiMuMu_*_*',
        'keep recoTrackExtras_ALCARECOTkAlJpsiMuMu_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlJpsiMuMu_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMu_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

# add branches for MC truth evaluation
from GeneratorInterface.Configuration.GeneratorInterface_EventContent_cff import GeneratorInterfaceAOD
from SimGeneral.Configuration.SimGeneral_EventContent_cff import SimGeneralAOD

OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands.extend(GeneratorInterfaceAOD.outputCommands)
_modifiedCommandsForGEN =  OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands.copy()
_modifiedCommandsForGEN.remove('keep *_genParticles_*_*')    # full genParticles list is too heavy
_modifiedCommandsForGEN.append('keep *_TkAlJpsiMuMuGenMuonSelector_*_*') # Keep only the filtered gen muons
OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands = _modifiedCommandsForGEN

OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands.extend(SimGeneralAOD.outputCommands)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlJpsiMuMu_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlJpsiMuMu_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlJpsiMuMu_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlJpsiMuMu_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlJpsiMuMu_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlJpsiMuMu = OutALCARECOTkAlJpsiMuMu_noDrop.clone()
OutALCARECOTkAlJpsiMuMu.outputCommands.insert(0, "drop *")
-- dummy change --

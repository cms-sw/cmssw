import FWCore.ParameterSet.Config as cms

# AlCaReco for track based alignment using UpsilonMuMu events
OutALCARECOTkAlUpsilonMuMu_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECOTkAlUpsilonMuMu')
    ),
    outputCommands = cms.untracked.vstring( 
        'keep recoTracks_ALCARECOTkAlUpsilonMuMu_*_*',
        'keep recoTrackExtras_ALCARECOTkAlUpsilonMuMu_*_*',
        'keep TrackingRecHitsOwned_ALCARECOTkAlUpsilonMuMu_*_*',
        'keep SiPixelClusteredmNewDetSetVector_ALCARECOTkAlUpsilonMuMu_*_*',
        'keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlUpsilonMuMu_*_*',
        'keep L1AcceptBunchCrossings_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_gtDigis_*_*',
        'keep *_TriggerResults_*_*',
        'keep DcsStatuss_scalersRawToDigi_*_*',
	'keep *_offlinePrimaryVertices_*_*')
)

# add branches for MC truth evaluation
from GeneratorInterface.Configuration.GeneratorInterface_EventContent_cff import GeneratorInterfaceAOD
from SimGeneral.Configuration.SimGeneral_EventContent_cff import SimGeneralAOD

OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands.extend(GeneratorInterfaceAOD.outputCommands)
_modifiedCommandsForGEN =  OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands.copy()
_modifiedCommandsForGEN.remove('keep *_genParticles_*_*')    # full genParticles list is too heavy
_modifiedCommandsForGEN.append('keep *_TkAlUpsilonMuMuGenMuonSelector_*_*') # Keep only the filtered gen muons
OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands = _modifiedCommandsForGEN

OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands.extend(SimGeneralAOD.outputCommands)

# in Run3, SCAL digis replaced by onlineMetaDataDigis
import copy
_run3_common_removedCommands = OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands.copy()
_run3_common_removedCommands.remove('keep DcsStatuss_scalersRawToDigi_*_*')

_run3_common_extraCommands = ['keep DCSRecord_onlineMetaDataDigis_*_*']

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(OutALCARECOTkAlUpsilonMuMu_noDrop, outputCommands = _run3_common_removedCommands + _run3_common_extraCommands)

# in Phase2, remove the SiStrip clusters and keep the OT ones instead
_phase2_common_removedCommands = OutALCARECOTkAlUpsilonMuMu_noDrop.outputCommands.copy()
_phase2_common_removedCommands.remove('keep SiStripClusteredmNewDetSetVector_ALCARECOTkAlUpsilonMuMu_*_*')

_phase2_common_extraCommands = ['keep Phase2TrackerCluster1DedmNewDetSetVector_ALCARECOTkAlUpsilonMuMu_*_*']

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(OutALCARECOTkAlUpsilonMuMu_noDrop, outputCommands = _phase2_common_removedCommands + _phase2_common_extraCommands )

OutALCARECOTkAlUpsilonMuMu = OutALCARECOTkAlUpsilonMuMu_noDrop.clone()
OutALCARECOTkAlUpsilonMuMu.outputCommands.insert(0, "drop *")

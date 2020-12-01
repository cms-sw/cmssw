import FWCore.ParameterSet.Config as cms

# Defines the L1 Emulator sequence for simulation use-case subsystem emulators
# run on the results of previous (in the hardware chain) subsystem emulator:
#  
#     SimL1Emulator = cms.Sequence(...)
#
# properly configured for the current Era (e.g. Run1, 2015, or 2016).  Also
# configures event setup producers appropriate to the current Era, to handle
# conditions which are not yet available in the GT.
#
# Author List
# Jim Brooke, 24 April 2008
# Vasile Mihai Ghete, 2009
# Jim Brooke, Michael Mulhearn, 2015
# Vladimir Rekovic 2016,2017

# Notes on Inputs:

# ECAL TPG emulator and HCAL TPG run in the simulation sequence in order to be able 
# to use unsuppressed digis produced by ECAL and HCAL simulation, respectively
# in Configuration/StandardSequences/python/Digi_cff.py
# SimCalorimetry.Configuration.SimCalorimetry_cff
# which calls
# SimCalorimetry.Configuration.ecalDigiSequence_cff
# SimCalorimetry.Configuration.hcalDigiSequence_cff

#
# At the moment, there is no emulator available for upgrade HF Trigger Primitives,
# so these missing (required!) inputs are presently ignored by downstream modules.
#

from L1Trigger.Configuration.SimL1TechnicalTriggers_cff import *

from L1Trigger.L1TCalorimeter.simDigis_cff import *
from L1Trigger.L1TMuon.simDigis_cff import *
from L1Trigger.L1TGlobal.simDigis_cff import *

# define a core which can be extented in customizations:
SimL1EmulatorCoreTask = cms.Task(
    SimL1TCalorimeterTask,
    SimL1TMuonTask,
    SimL1TechnicalTriggersTask,
    SimL1TGlobalTask
)
SimL1EmulatorCore = cms.Sequence(SimL1EmulatorCoreTask)

SimL1EmulatorTask = cms.Task(SimL1EmulatorCoreTask)
SimL1Emulator = cms.Sequence( SimL1EmulatorTask )

# 
# Emulators are configured from DB (GlobalTags)
#

from L1Trigger.L1TGlobal.GlobalParameters_cff import *

# 2017 EMTF and TwinMux emulator use payloads from DB, not yet in GT,
# soon to be removed when availble in GTs
from L1Trigger.L1TTwinMux.fakeTwinMuxParams_cff import *

_phase2_siml1emulator = SimL1EmulatorTask.copy()

# ########################################################################
# ########################################################################
#
# Phase-2 
#
# ########################################################################
# ########################################################################

# ########################################################################
# Phase-2 Trigger Primitives
# ########################################################################

# HGCAL TP 
# ########################################################################
from  L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
_phase2_siml1emulator.add(hgcalTriggerPrimitivesTask)
 
# ########################################################################
# Phase 2 L1T
# ########################################################################

# Barrel and EndCap EGamma
# ########################################################################

from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *
_phase2_siml1emulator.add(L1EGammaClusterEmuProducer)

from L1Trigger.L1CaloTrigger.l1EGammaEEProducer_cfi import *
_phase2_siml1emulator.add(l1EGammaEEProducer)

# ########################################################################
# Phase-2 L1T - TrackTrigger dependent modules
# ########################################################################

# Tk + StandaloneObj, including L1TkPrimaryVertex
# ########################################################################

from L1Trigger.L1TTrackMatch.L1TkPrimaryVertexProducer_cfi import L1TkPrimaryVertex
from L1Trigger.L1TTrackMatch.L1TkElectronTrackProducer_cfi import L1TkElectronsCrystal, L1TkElectronsLooseCrystal, L1TkElectronsEllipticMatchCrystal, L1TkIsoElectronsCrystal, L1TkElectronsHGC, L1TkElectronsEllipticMatchHGC, L1TkIsoElectronsHGC
from L1Trigger.L1TTrackMatch.L1TkEmParticleProducer_cfi import L1TkPhotonsCrystal, L1TkPhotonsHGC
from L1Trigger.L1TTrackMatch.L1TkMuonProducer_cfi import L1TkMuons

_phase2_siml1emulator.add(L1TkPrimaryVertex)

_phase2_siml1emulator.add(L1TkElectronsCrystal)
_phase2_siml1emulator.add(L1TkElectronsLooseCrystal)
_phase2_siml1emulator.add(L1TkElectronsEllipticMatchCrystal)
_phase2_siml1emulator.add(L1TkIsoElectronsCrystal)
_phase2_siml1emulator.add(L1TkPhotonsCrystal)

_phase2_siml1emulator.add(L1TkElectronsHGC)
_phase2_siml1emulator.add(L1TkElectronsEllipticMatchHGC)
_phase2_siml1emulator.add(L1TkIsoElectronsHGC)
_phase2_siml1emulator.add(L1TkPhotonsHGC)

_phase2_siml1emulator.add( L1TkMuons )

# PF Candidates
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff import *
_phase2_siml1emulator.add(l1ParticleFlowTask)

# PF Jet
# ########################################################################
from L1Trigger.L1CaloTrigger.Phase1L1TJets_cff import *
# Describe here l1PFJets_a_la_Phase1 Task
# ###############################
l1PFJetsPhase1Task = cms.Task(Phase1L1TJetProducer , Phase1L1TJetCalibrator)
_phase2_siml1emulator.add(l1PFJetsPhase1Task)

# PF MET
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import *
# Describe here l1PFMets Task
# ###############################
l1PFMetsTask = cms.Task(l1PFMetCalo , l1PFMetPF , l1PFMetPuppi)
_phase2_siml1emulator.add(l1PFMetsTask)

# --> add modules
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith( SimL1EmulatorTask , _phase2_siml1emulator)

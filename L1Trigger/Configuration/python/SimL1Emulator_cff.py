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
from L1Trigger.DTTriggerPhase2.CalibratedDigis_cfi import *
CalibratedDigis.dtDigiTag = "simMuonDTDigis"
_phase2_siml1emulator.add(CalibratedDigis)
from L1Trigger.DTTriggerPhase2.dtTriggerPhase2PrimitiveDigis_cfi import *
_phase2_siml1emulator.add(dtTriggerPhase2PrimitiveDigis)

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

# Barrel and EndCap CaloJet/HT
# ########################################################################
# ----    Produce the calibrated tower collection combining Barrel, HGCal, HF
from L1Trigger.L1CaloTrigger.L1TowerCalibrationProducer_cfi import *
L1TowerCalibration = L1TowerCalibrationProducer.clone(
  L1HgcalTowersInputTag = ("hgcalTowerProducer","HGCalTowerProcessor",""),
  l1CaloTowers = ("L1EGammaClusterEmuProducer","L1CaloTowerCollection","")
)
# ----    Produce the L1CaloJets
from L1Trigger.L1CaloTrigger.L1CaloJetProducer_cfi import *
L1CaloJet = L1CaloJetProducer.clone (
    l1CaloTowers = ("L1TowerCalibration","L1CaloTowerCalibratedCollection",""),
    L1CrystalClustersInputTag = ("L1EGammaClusterEmuProducer", "","")
)
# ----    Produce the CaloJet HTT Sums
from L1Trigger.L1CaloTrigger.L1CaloJetHTTProducer_cfi import *
L1CaloJetHTT = L1CaloJetHTTProducer.clone(
    BXVCaloJetsInputTag = ("L1CaloJet", "CaloJets") 
)


_phase2_siml1emulator.add(L1TowerCalibration)
_phase2_siml1emulator.add(L1CaloJet)
_phase2_siml1emulator.add(L1CaloJetHTT)

# ########################################################################
# Phase-2 L1T - TrackTrigger dependent modules
# ########################################################################
from L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi import *
from L1Trigger.VertexFinder.VertexProducer_cff import *
L1VertexFinder = VertexProducer.clone()
L1VertexFinderEmulator = VertexProducer.clone()
L1VertexFinderEmulator.VertexReconstruction.Algorithm = "fastHistoEmulation"
L1VertexFinderEmulator.l1TracksInputTag = ("L1GTTInputProducer","Level1TTTracksConverted")
_phase2_siml1emulator.add(L1VertexFinder)
_phase2_siml1emulator.add(L1GTTInputProducer)
_phase2_siml1emulator.add(L1GTTInputProducerExtended)
_phase2_siml1emulator.add(L1VertexFinderEmulator)

# Emulated GMT Muons (Tk + Stub, Tk + MuonTFT, StandaloneMuon)
# ########################################################################
from L1Trigger.Phase2L1GMT.gmt_cfi  import *
L1TkStubsGmt = gmtStubs.clone()
L1TkMuonsGmt = gmtMuons.clone(
    srcStubs  = 'L1TkStubsGmt'
)
L1SAMuonsGmt = standaloneMuons.clone()
_phase2_siml1emulator.add( L1TkStubsGmt )
_phase2_siml1emulator.add( L1TkMuonsGmt )
_phase2_siml1emulator.add( L1SAMuonsGmt )

# Tracker Objects
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TrackJetProducer_cfi import *
from L1Trigger.L1TTrackMatch.L1TrackFastJetProducer_cfi import *
from L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi import *
from L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi import *
# make the input tags consistent with the choice L1VertexFinder above
L1TrackJets.L1PVertexCollection  = ("L1VertexFinder", L1VertexFinder.l1VertexCollectionName.value())
L1TrackJetsExtended.L1PVertexCollection  = ("L1VertexFinder", L1VertexFinder.l1VertexCollectionName.value())
#L1TrackerEtMiss.L1VertexInputTag = ("L1VertexFinder", L1VertexFinder.l1VertexCollectionName.value())
#L1TrackerEtMissExtended.L1VertexInputTag = ("L1VertexFinder", L1VertexFinder.l1VertexCollectionName.value())
_phase2_siml1emulator.add(L1TrackJets)
_phase2_siml1emulator.add(L1TrackJetsExtended)
_phase2_siml1emulator.add(L1TrackFastJets)

_phase2_siml1emulator.add(L1TrackerEtMiss)
_phase2_siml1emulator.add(L1TrackerHTMiss)

#Emulated tracker objects
from L1Trigger.L1TTrackMatch.L1TrackJetEmulationProducer_cfi import *
_phase2_siml1emulator.add(L1TrackJetsEmulation)
_phase2_siml1emulator.add(L1TrackJetsExtendedEmulation)

from L1Trigger.L1TTrackMatch.L1TrackerEtMissEmulatorProducer_cfi import *
L1TrackerEmuEtMiss.L1VertexInputTag = ("L1VertexFinderEmulator","l1verticesEmulation")
_phase2_siml1emulator.add(L1TrackerEmuEtMiss)

from L1Trigger.L1TTrackMatch.L1TkHTMissEmulatorProducer_cfi import *
_phase2_siml1emulator.add(L1TrackerEmuHTMiss)
_phase2_siml1emulator.add(L1TrackerEmuHTMissExtended)

# PF Candidates
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1ctLayer1_cff import *
from L1Trigger.Phase2L1ParticleFlow.l1ctLayer2EG_cff import *
_phase2_siml1emulator.add(l1ctLayer1TaskInputsTask, l1ctLayer1Task, l1ctLayer2EGTask)

# PF Jet
# ########################################################################
from L1Trigger.L1CaloTrigger.Phase1L1TJets_cff import *
# Describe here l1PFJets_a_la_Phase1 Task
# ###############################
l1PFJetsPhase1Task = cms.Task(Phase1L1TJetProducer , Phase1L1TJetCalibrator, Phase1L1TJetSumsProducer)
_phase2_siml1emulator.add(l1PFJetsPhase1Task)

from L1Trigger.Phase2L1Taus.HPSPFTauProducerPF_cfi import *
_phase2_siml1emulator.add(HPSPFTauProducerPF)

from L1Trigger.Phase2L1Taus.HPSPFTauProducerPuppi_cfi import *
_phase2_siml1emulator.add(HPSPFTauProducerPuppi)

from L1Trigger.L1CaloTrigger.Phase1L1TJets_9x9_cff import *
l1PFJetsPhase1Task_9x9 = cms.Task(  Phase1L1TJetProducer9x9, Phase1L1TJetCalibrator9x9, Phase1L1TJetSumsProducer9x9)
_phase2_siml1emulator.add(l1PFJetsPhase1Task_9x9)


# PF MET
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import *
_phase2_siml1emulator.add(l1PFJetsTask)

from L1Trigger.Phase2L1ParticleFlow.L1MetPfProducer_cfi import *
_phase2_siml1emulator.add(L1MetPfProducer)


# NNTaus
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.L1NNTauProducer_cff import *
_phase2_siml1emulator.add(L1NNTauProducerPuppi)

# --> add modules
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith( SimL1EmulatorTask , _phase2_siml1emulator)

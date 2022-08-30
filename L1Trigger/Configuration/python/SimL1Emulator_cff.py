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
_phase2_siml1emulator.add(l1tEGammaClusterEmuProducer)

# Barrel and EndCap CaloJet/HT
# ########################################################################
# ----    Produce the calibrated tower collection combining Barrel, HGCal, HF
from L1Trigger.L1CaloTrigger.L1TowerCalibrationProducer_cfi import *
l1tTowerCalibration = l1tTowerCalibrationProducer.clone(
  L1HgcalTowersInputTag = ("hgcalTowerProducer","HGCalTowerProcessor",""),
  l1CaloTowers = ("l1tEGammaClusterEmuProducer","L1CaloTowerCollection","")
)
# ----    Produce the L1CaloJets
from L1Trigger.L1CaloTrigger.L1CaloJetProducer_cfi import *
l1tCaloJet = l1tCaloJetProducer.clone (
    l1CaloTowers = ("l1tTowerCalibration","L1CaloTowerCalibratedCollection",""),
    L1CrystalClustersInputTag = ("l1tEGammaClusterEmuProducer", "","")
)
# ----    Produce the CaloJet HTT Sums
from L1Trigger.L1CaloTrigger.L1CaloJetHTTProducer_cfi import *
l1tCaloJetHTT = l1tCaloJetHTTProducer.clone(
    BXVCaloJetsInputTag = ("L1CaloJet", "CaloJets") 
)


_phase2_siml1emulator.add(l1tTowerCalibration)
_phase2_siml1emulator.add(l1tCaloJet)
_phase2_siml1emulator.add(l1tCaloJetHTT)

# ########################################################################
# Phase-2 L1T - TrackTrigger dependent modules
# ########################################################################
from L1Trigger.L1TTrackMatch.L1GTTInputProducer_cfi import *
from L1Trigger.VertexFinder.VertexProducer_cff import *
l1tVertexFinder = l1tVertexProducer.clone()
l1tVertexFinderEmulator = l1tVertexProducer.clone()
l1tVertexFinderEmulator.VertexReconstruction.Algorithm = "fastHistoEmulation"
l1tVertexFinderEmulator.l1TracksInputTag = ("l1tGTTInputProducer","Level1TTTracksConverted")
_phase2_siml1emulator.add(l1tVertexFinder)
_phase2_siml1emulator.add(l1tGTTInputProducer)
_phase2_siml1emulator.add(l1tGTTInputProducerExtended)
_phase2_siml1emulator.add(l1tVertexFinderEmulator)

# Emulated GMT Muons (Tk + Stub, Tk + MuonTFT, StandaloneMuon)
# ########################################################################
from L1Trigger.Phase2L1GMT.gmt_cfi  import *
l1tTkStubsGmt = l1tGMTStubs.clone()
l1tTkMuonsGmt = l1tGMTMuons.clone(
    srcStubs  = 'l1tTkStubsGmt'
)
l1tSAMuonsGmt = l1tStandaloneMuons.clone()
_phase2_siml1emulator.add( l1tTkStubsGmt )
_phase2_siml1emulator.add( l1tTkMuonsGmt )
_phase2_siml1emulator.add( l1tSAMuonsGmt )

# Tracker Objects
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TrackJetProducer_cfi import *
from L1Trigger.L1TTrackMatch.L1TrackFastJetProducer_cfi import *
from L1Trigger.L1TTrackMatch.L1TrackerEtMissProducer_cfi import *
from L1Trigger.L1TTrackMatch.L1TkHTMissProducer_cfi import *
# make the input tags consistent with the choice L1VertexFinder above
l1tTrackJets.L1PVertexCollection  = ("L1VertexFinder", l1tVertexFinder.l1VertexCollectionName.value())
l1tTrackJetsExtended.L1PVertexCollection  = ("L1VertexFinder", l1tVertexFinder.l1VertexCollectionName.value())
#L1TrackerEtMiss.L1VertexInputTag = ("L1VertexFinder", L1VertexFinder.l1VertexCollectionName.value())
#L1TrackerEtMissExtended.L1VertexInputTag = ("L1VertexFinder", L1VertexFinder.l1VertexCollectionName.value())
_phase2_siml1emulator.add(l1tTrackJets)
_phase2_siml1emulator.add(l1tTrackJetsExtended)
_phase2_siml1emulator.add(l1tTrackFastJets)

_phase2_siml1emulator.add(l1tTrackerEtMiss)
_phase2_siml1emulator.add(l1tTrackerHTMiss)

#Emulated tracker objects
from L1Trigger.L1TTrackMatch.L1TrackJetEmulationProducer_cfi import *
_phase2_siml1emulator.add(l1tTrackJetsEmulation)
_phase2_siml1emulator.add(l1tTrackJetsExtendedEmulation)

from L1Trigger.L1TTrackMatch.L1TrackerEtMissEmulatorProducer_cfi import *
l1tTrackerEmuEtMiss.L1VertexInputTag = ("L1VertexFinderEmulator","l1verticesEmulation")
_phase2_siml1emulator.add(l1tTrackerEmuEtMiss)

from L1Trigger.L1TTrackMatch.L1TkHTMissEmulatorProducer_cfi import *
_phase2_siml1emulator.add(l1tTrackerEmuHTMiss)
_phase2_siml1emulator.add(l1tTrackerEmuHTMissExtended)

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
l1tPFJetsPhase1Task = cms.Task(l1tPhase1JetProducer , l1tPhase1JetCalibrator, l1tPhase1JetSumsProducer)
_phase2_siml1emulator.add(l1tPFJetsPhase1Task)

from L1Trigger.Phase2L1Taus.HPSPFTauProducerPF_cfi import *
_phase2_siml1emulator.add(l1tHPSPFTauProducerPF)

from L1Trigger.Phase2L1Taus.HPSPFTauProducerPuppi_cfi import *
_phase2_siml1emulator.add(l1tHPSPFTauProducerPuppi)

from L1Trigger.L1CaloTrigger.Phase1L1TJets_9x9_cff import *
l1tPFJetsPhase1Task_9x9 = cms.Task(  l1tPhase1JetProducer9x9, l1tPhase1JetCalibrator9x9, l1tPhase1JetSumsProducer9x9)
_phase2_siml1emulator.add(l1tPFJetsPhase1Task_9x9)


# PF MET
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import *
_phase2_siml1emulator.add(l1tPFJetsTask)

from L1Trigger.Phase2L1ParticleFlow.L1MetPfProducer_cfi import *
_phase2_siml1emulator.add(l1tMETPFProducer)


# NNTaus
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.L1NNTauProducer_cff import *
_phase2_siml1emulator.add(l1tNNTauProducerPuppi)

# --> add modules
from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith( SimL1EmulatorTask , _phase2_siml1emulator)

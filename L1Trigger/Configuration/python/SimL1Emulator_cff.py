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
SimL1EmulatorCore = cms.Sequence(
    SimL1TCalorimeter +
    SimL1TMuon +
    SimL1TechnicalTriggers +
    SimL1TGlobal
    )

SimL1Emulator = cms.Sequence( SimL1EmulatorCore )

#
# Emulators are configured from DB (GlobalTags)
#

from L1Trigger.L1TGlobal.GlobalParameters_cff import *

# 2017 EMTF and TwinMux emulator use payloads from DB, not yet in GT,
# soon to be removed when availble in GTs
from L1Trigger.L1TTwinMux.fakeTwinMuxParams_cff import *

# ########################################################################
# Customisation for the phase2_hgcal era. Includes the HGCAL TPs
# ########################################################################
from  L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
_phase2_siml1emulator = SimL1Emulator.copy()
_phase2_siml1emulator += hgcalTriggerPrimitives

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith( SimL1Emulator , _phase2_siml1emulator )


# ########################################################################
# Customisation for the phase2_ecal era. Includes the ecalEB TPs
# ########################################################################
from SimCalorimetry.EcalEBTrigPrimProducers.ecalEBTriggerPrimitiveDigis_cff import *
_phase2_siml1emulator_ebtp = SimL1Emulator.copy()
_phase2_siml1emulator_ebtp += simEcalEBTriggerPrimitiveDigis

from Configuration.Eras.Modifier_phase2_ecal_cff import phase2_ecal
phase2_ecal.toReplaceWith( SimL1Emulator , _phase2_siml1emulator_ebtp )

# If PreMixing, don't run these modules during first step
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toReplaceWith(SimL1Emulator, SimL1Emulator.copyAndExclude([
    SimL1TCalorimeter,
    SimL1TechnicalTriggers,
    SimL1TGlobal
]))

# ########################################################################
# Customisation for the phase2_trigger era, assumes TrackTrigger available
# ########################################################################
phase2_SimL1Emulator = SimL1Emulator.copy()

# Vertex
# ########################################################################
from L1Trigger.VertexFinder.VertexProducer_cff import *

phase2_SimL1Emulator += VertexProducer

# Barrel EGamma
# ########################################################################
from L1Trigger.L1CaloTrigger.l1EGammaCrystalsProducer_cfi import *
phase2_SimL1Emulator += l1EGammaCrystalsProducer
from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *
phase2_SimL1Emulator += L1EGammaClusterEmuProducer

from L1Trigger.L1CaloTrigger.l1EGammaEEProducer_cfi import *
phase2_SimL1Emulator += l1EGammaEEProducer

#  CaloJets
# ########################################################################
from L1Trigger.L1CaloTrigger.L1CaloJets_cff import *
phase2_SimL1Emulator += l1CaloJetsSequence

# Barrel L1Tk + Stub
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TTrackerPlusStubs_cfi import *
l1KBmtfStubMatchedMuons = l1StubMatchedMuons.clone()
phase2_SimL1Emulator += l1KBmtfStubMatchedMuons
# EndCap L1Tk + Stub
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TkMuonStubProducer_cfi import *
l1TkMuonStubEndCap = L1TkMuonStub.clone()
phase2_SimL1Emulator += l1TkMuonStubEndCap
l1TkMuonStubEndCapS12 = L1TkMuonStubS12.clone()
phase2_SimL1Emulator += l1TkMuonStubEndCapS12

# Tk + StandaloneObj
# (include L1TkPrimaryVertex)
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TkObjectProducers_cff import *
phase2_SimL1Emulator += L1TkPrimaryVertex
#phase2_SimL1Emulator += L1TkElectrons # warning this has a PhaseI EG seed!
#phase2_SimL1Emulator += L1TkIsoElectrons # warning this has a PhaseI EG seed!
#phase2_SimL1Emulator += L1TkPhotons # warning this has a PhaseI EG seed!
phase2_SimL1Emulator += L1TkElectronsCrystal
phase2_SimL1Emulator += L1TkIsoElectronsCrystal
phase2_SimL1Emulator += L1TkElectronsLooseCrystal
phase2_SimL1Emulator += L1TkPhotonsCrystal
phase2_SimL1Emulator += L1TkElectronsHGC
phase2_SimL1Emulator += L1TkIsoElectronsHGC
phase2_SimL1Emulator += L1TkElectronsLooseHGC
phase2_SimL1Emulator += L1TkPhotonsHGC

phase2_SimL1Emulator += L1TkCaloJets
phase2_SimL1Emulator += TwoLayerJets
phase2_SimL1Emulator += L1TrackerJets
phase2_SimL1Emulator += L1TrackerEtMiss
phase2_SimL1Emulator += L1TkCaloHTMissVtx
phase2_SimL1Emulator += L1TrackerHTMiss
phase2_SimL1Emulator += L1TkMuons
phase2_SimL1Emulator += L1TkMuonsTP
phase2_SimL1Emulator += L1TkGlbMuons
phase2_SimL1Emulator += L1TkTauFromCalo
phase2_SimL1Emulator += L1TrackerTaus
phase2_SimL1Emulator += L1TkEGTaus
phase2_SimL1Emulator += L1TkCaloTaus


# PF Candidates
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff import *
phase2_SimL1Emulator += l1ParticleFlow

from L1Trigger.Phase2L1ParticleFlow.l1pfJetMet_cff import *
# Describe here l1PFJets sequence
# ###############################
#l1PFJets = cms.Sequence(
#  ak4PFL1Calo + ak4PFL1PF + ak4PFL1Puppi +
#  ak4PFL1CaloCorrected + ak4PFL1PFCorrected + ak4PFL1PuppiCorrected)
phase2_SimL1Emulator += l1PFJets
# Describe here l1PFMets sequence
# ###############################
#l1PFMets = cms.Sequence(l1PFMetCalo + l1PFMetPF + l1PFMetPuppi)
phase2_SimL1Emulator += l1PFMets

# PFTaus(HPS)
# ########################################################################
from L1Trigger.Phase2L1Taus.L1PFTauProducer_cff import L1PFTauProducer
l1pfTauProducer = L1PFTauProducer.clone()
l1pfTauProducer.L1PFObjects = cms.InputTag("l1pfCandidates","PF")
l1pfTauProducer.L1Neutrals = cms.InputTag("l1pfCandidates")
phase2_SimL1Emulator += l1pfTauProducer

# NNTaus
# ########################################################################
from L1Trigger.Phase2L1Taus.L1NNTauProducer_cff import *
l1NNTauProducer = L1NNTauProducer.clone()
l1NNTauProducer.L1PFObjects = cms.InputTag("l1pfCandidates","PF")
l1NNTauProducerPuppi = L1NNTauProducerPuppi.clone()
l1NNTauProducerPuppi.L1PFObjects = cms.InputTag("l1pfCandidates","PF")
phase2_SimL1Emulator += l1NNTauProducer
phase2_SimL1Emulator += l1NNTauProducerPuppi

# NNTaus
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TkBsCandidateProducer_cfi import *
l1TkBsCandidates = L1TkBsCandidates.clone()
l1TkBsCandidatesLooseWP = L1TkBsCandidatesLooseWP.clone()
l1TkBsCandidatesTightWP = L1TkBsCandidatesTightWP.clone()
phase2_SimL1Emulator += l1TkBsCandidates
phase2_SimL1Emulator += l1TkBsCandidatesLooseWP
phase2_SimL1Emulator += l1TkBsCandidatesTightWP

from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith( SimL1Emulator , phase2_SimL1Emulator)

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

# ########################################################################
# Customisation for the phase2_hgcal era. Includes the HGCAL TPs
# ########################################################################
from  L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
_phase2_siml1emulator = SimL1EmulatorTask.copy()
_phase2_siml1emulator.add(hgcalTriggerPrimitivesTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
(phase2_hgcal & ~phase2_hgcalV9).toReplaceWith( SimL1Emulator , _phase2_siml1emulator )

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

# Kalman Barrel MTF
# ########################################################################
from L1Trigger.L1TMuonBarrel.simKBmtfStubs_cfi import *
from L1Trigger.L1TMuonBarrel.simKBmtfDigis_cfi import *

phase2_SimL1Emulator += simKBmtfStubs
phase2_SimL1Emulator += simKBmtfDigis

# Barrel EGamma
# ########################################################################
from L1Trigger.L1CaloTrigger.l1EGammaCrystalsProducer_cfi import *

phase2_SimL1Emulator += l1EGammaCrystalsProducer

# Tk + StandaloneObj
# (include L1TkPrimaryVertex)
# ########################################################################
from L1Trigger.L1TTrackMatch.L1TkObjectProducers_cff import *
phase2_SimL1Emulator += L1TkElectrons
phase2_SimL1Emulator += L1TkIsoElectrons
phase2_SimL1Emulator += L1TkPhotons
phase2_SimL1Emulator += L1TkPrimaryVertex
phase2_SimL1Emulator += L1TkCaloJets
phase2_SimL1Emulator += L1TrackerJets
phase2_SimL1Emulator += L1TrackerEtMiss
phase2_SimL1Emulator += L1TkCaloHTMissVtx
phase2_SimL1Emulator += L1TrackerHTMiss
phase2_SimL1Emulator += L1TkMuons
phase2_SimL1Emulator += L1TkTauFromCalo

# PF Candidates
# ########################################################################
from L1Trigger.Phase2L1ParticleFlow.l1ParticleFlow_cff import *
#l1ParticleFlow = cms.Sequence(
#    l1EGammaCrystalsProducer +
#    pfTracksFromL1Tracks +
#    pfClustersFromHGC3DClustersEM +
#    pfClustersFromL1EGClusters +
#    pfClustersFromCombinedCalo +
#    l1pfProducer
#)
l1pfProducerTightTK = l1pfProducer.clone(trkMinStubs = 6)
l1ParticleFlow += l1pfProducerTightTK

phase2_SimL1Emulator += l1ParticleFlow

# PF METs
# ########################################################################
from RecoMET.METProducers.PFMET_cfi import pfMet
pfMet.calculateSignificance = False
l1MetCalo    = pfMet.clone(src = "l1pfProducer:Calo")
l1MetTK      = pfMet.clone(src = "l1pfProducer:TK")
l1MetTKV     = pfMet.clone(src = "l1pfProducer:TKVtx")
l1MetTightTK      = pfMet.clone(src = "l1pfProducerTightTK:TK")
l1MetTightTKV     = pfMet.clone(src = "l1pfProducerTightTK:TKVtx")
l1MetPF      = pfMet.clone(src = "l1pfProducer:PF")
l1MetPuppi   = pfMet.clone(src = "l1pfProducer:Puppi")
l1PFMets = cms.Sequence( l1MetCalo + l1MetTK + l1MetTKV + l1MetPF + l1MetPuppi
                        + l1MetTightTK + l1MetTightTKV)

phase2_SimL1Emulator += l1PFMets

# PF Jets
# ########################################################################
from RecoJets.JetProducers.ak4PFJets_cfi import ak4PFJets
ak4L1Calo    = ak4PFJets.clone(src = 'l1pfProducer:Calo')
ak4L1TK      = ak4PFJets.clone(src = 'l1pfProducer:TK')
ak4L1TKV     = ak4PFJets.clone(src = 'l1pfProducer:TKVtx')
ak4L1TightTK      = ak4PFJets.clone(src = 'l1pfProducerTightTK:TK')
ak4L1TightTKV     = ak4PFJets.clone(src = 'l1pfProducerTightTK:TKVtx')
ak4L1PF      = ak4PFJets.clone(src = 'l1pfProducer:PF')
ak4L1Puppi   = ak4PFJets.clone(src = 'l1pfProducer:Puppi')
l1PFJets = cms.Sequence( ak4L1Calo + ak4L1TK + ak4L1TKV + ak4L1PF + ak4L1Puppi
                        + ak4L1TightTK + ak4L1TightTKV)

phase2_SimL1Emulator += l1PFJets

from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith( SimL1Emulator , phase2_SimL1Emulator)

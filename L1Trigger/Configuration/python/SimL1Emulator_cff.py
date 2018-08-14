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

# Customisation for the phase2_hgcal era. Includes the HGCAL L1 trigger
from  L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff import *
_phase2_siml1emulator = SimL1Emulator.copy()
_phase2_siml1emulator += hgcalTriggerPrimitives

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
(phase2_hgcal & ~phase2_hgcalV9).toReplaceWith( SimL1Emulator , _phase2_siml1emulator )

# If PreMixing, don't run these modules during first step
# TODO: Do we actually need anything from here run in stage1?
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
premix_stage1.toReplaceWith(SimL1Emulator, SimL1Emulator.copyAndExclude([
    SimL1TCalorimeter,
    SimL1TechnicalTriggers,
    SimL1TGlobal
]))

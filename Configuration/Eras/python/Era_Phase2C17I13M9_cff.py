import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C11I13M9_cff import Phase2C11I13M9
from Configuration.Eras.Modifier_phase2_hgcalV12_cff import phase2_hgcalV12
from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16
# MC-truth graph on by default for Run4. This is the common HGCal-geometry-agnostic
# Phase-2 base, so every geometry era layered on it (C20 hfnose, C22 V18, C26 V19 and
# future versions) inherits truth. Gated at DIGI by (enableTruth & phase2_hgcal);
# FastSim drops it via fastSimPhase2, premix drops it at the digitizer level.
from Configuration.ProcessModifiers.enableTruth_cff import enableTruth

Phase2C17I13M9 = cms.ModifierChain(Phase2C11I13M9.copyAndExclude([phase2_hgcalV12]), phase2_hgcalV16, enableTruth)

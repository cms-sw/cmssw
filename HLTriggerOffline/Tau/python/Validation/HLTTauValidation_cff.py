import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *
from Validation.RecoTau.hltTauValidation_cff import *

HLTTauVal       = cms.Sequence(hltTauRef+hltTauValIdeal)
HLTTauValPhase2 = cms.Sequence(hltTauValidation+hltTauValidation_deltaR0p1+hltTauValidation_deltaR0p15+hltTauValidation_deltaR0p2+hltTauValidation_deltaR0p25)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toReplaceWith(HLTTauVal, HLTTauValPhase2)
import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_8E29_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_1E31_cfi import *


HLTTauVal    = cms.Sequence(hltTauRef+hltTauValDefault+hltTauValStartup+hltTauValIdeal)





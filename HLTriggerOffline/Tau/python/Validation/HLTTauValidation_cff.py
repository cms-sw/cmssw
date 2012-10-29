import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi import *
#from HLTriggerOffline.Tau.Validation.HLTTauValidation_8E29_cfi import *
#from HLTriggerOffline.Tau.Validation.HLTTauValidation_1E31_cfi import *
#from HLTriggerOffline.Tau.Validation.HLTTauValidation_6E31_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_5E32_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_1E33_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_3E33_cfi import *

HLTTauVal    = cms.Sequence(hltTauRef+hltTauValIdeal5E32+hltTauValIdeal1E33+hltTauValIdeal3E33)

# for FS, hltTauRef Producers go into separate "prevalidation" sequence
# (this fixes the "no EDProducer in EndPath" problem)
HLTTauValFS  = cms.Sequence(hltTauValIdeal5E32+hltTauValIdeal1E33+hltTauValIdeal3E33)




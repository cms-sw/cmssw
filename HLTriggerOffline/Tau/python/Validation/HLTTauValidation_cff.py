import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.Validation.HLTTauReferences_cfi import *
#from HLTriggerOffline.Tau.Validation.DoubleTauValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.SingleTauValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.SingleTauMETValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.ElectronTauValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.MuonTauValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.L1TauValidation_cfi import *
from HLTriggerOffline.Tau.Validation.HLTTauValidation_cfi import *

HLTTauVal    = cms.Sequence(HLTTauRef+HLTTauValidationSequence)





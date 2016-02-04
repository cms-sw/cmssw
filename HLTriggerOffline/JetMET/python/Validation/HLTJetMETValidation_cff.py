import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.JetMET.Validation.HLTJetMETReferences_cfi import *
#from HLTriggerOffline.Tau.Validation.DoubleTauValidation_cfi import *
from HLTriggerOffline.JetMET.Validation.SingleJetValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.SingleTauMETValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.ElectronTauValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.MuonTauValidation_cfi import *
#from HLTriggerOffline.Tau.Validation.L1TauValidation_cfi import *

#HLTTauVal    = cms.Path(HLTTauRef+DoubleTauValidation+SingleTauValidation+SingleTauMETValidation+ElectronTauValidation+MuonTauValidation+L1TauVal)
# HLTJetMETVal    = cms.Path(HLTJetMETRef + SingleJetValidation)
#HLTJetMETVal    = cms.Path(SingleJetValidation)
##please do NOT include paths here!
HLTJetMETValSeq    = cms.Sequence(SingleJetValidation)
#HLTJetMETVal    = cms.Sequence(HLTJetMETRef + SingleJetValidation)



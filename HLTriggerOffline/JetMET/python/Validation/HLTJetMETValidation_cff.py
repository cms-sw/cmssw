import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.JetMET.Validation.SingleJetValidation_cfi import *

HLTJetMETValSeq    = cms.Sequence(SingleJetValidation)


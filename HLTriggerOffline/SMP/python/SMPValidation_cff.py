import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.SMP.hltSMPValidator_cfi import *

SMPValidationSequence = cms.Sequence(
    hltSMPValidator
    )


### This file defines the ExoticaValidationSequence, to be put
### together with the other sequences in
### HLTriggerOffline/Common/python/HLTValidation_cff.py

import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Exotica.hltExoticaValidator_cfi import *

ExoticaValidationSequence = cms.Sequence(
    hltExoticaValidator
    )

#HLTExoticaVal_FastSim = cms.Sequence(
#    recoExoticaValidationHLTFastSim_seq +
#    hltExoticaValidator
#    )

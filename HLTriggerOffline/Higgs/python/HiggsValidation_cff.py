import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Higgs.hltHiggsValidator_cfi import *

HiggsValidationSequence = cms.Sequence(
    hltHiggsValidator
    )

#HLTHiggsVal_FastSim = cms.Sequence(
#    recoHiggsValidationHLTFastSim_seq +
#    hltHiggsValidator
#    )

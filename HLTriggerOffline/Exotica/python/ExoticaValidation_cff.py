### This file defines the ExoticaValidationSequence, to be put
### together with the other sequences in
### HLTriggerOffline/Common/python/HLTValidation_cff.py
### Also defines some Producers.

import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Exotica.hltExoticaValidator_cfi import *

# We create a basic HT variable here
recoExoticaValidationHT = cms.EDProducer("PFMETProducer",
                                         src = cms.InputTag("ak5PFJetsCHS"),
                                         alias = cms.string('PFMHT'),
                                         globalThreshold = cms.double(30.0),
                                         calculateSignificance = cms.bool(False),
                                         jets = cms.InputTag("ak5PFJetsCHS") # for significance calculation
                                         )

ExoticaValidationProdSeq = cms.Sequence(
    recoExoticaValidationHT
    )

ExoticaValidationSequence = cms.Sequence(
    hltExoticaValidator
    )

#HLTExoticaVal_FastSim = cms.Sequence(
#    recoExoticaValidationHLTFastSim_seq +
#    hltExoticaValidator
#    )

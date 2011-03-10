import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFProducer.pfPhotonInterestingDetIds_cfi import *
from RecoParticleFlow.PFProducer.pfBasedPhotonIso_cff import * 
from RecoParticleFlow.PFProducer.pfPhotonTranslator_cfi import *

pfPhotonTranslatorSequence = cms.Sequence(
    pfBasedPhotonIsoSequence+
    pfPhotonTranslator+
    pfPhotonInterestingEcalDetIdEB+
    pfPhotonInterestingEcalDetIdEE
    )

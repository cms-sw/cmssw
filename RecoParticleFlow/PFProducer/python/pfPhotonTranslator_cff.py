import FWCore.ParameterSet.Config as cms
#from RecoParticleFlow.PFProducer.pfPhotonTranslator_cfi import *
#from RecoParticleFlow.PFProducer.pfPhotonInterestingDetIds_cfi import *
from RecoParticleFlow.PFProducer.pfBasedPhotonIso_cff import * 

pfPhotonTranslatorSequence = cms.Sequence(
    pfBasedPhotonIsoSequence
#    pfPhotonTranslator+
#    pfPhotonInterestingEcalDetIdEB+
#    pfPhotonInterestingEcalDetIdEE
    )

import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFProducer.pfPhotonInterestingDetIds_cfi import *
from RecoParticleFlow.PFProducer.pfBasedElectronPhotonIso_cff import * 
from RecoParticleFlow.PFProducer.pfPhotonTranslator_cfi import *

pfPhotonTranslatorSequence = cms.Sequence(
    pfBasedElectronPhotonIsoSequence+
    pfPhotonTranslator+
    pfPhotonInterestingEcalDetIdEB+
    pfPhotonInterestingEcalDetIdEE
    )

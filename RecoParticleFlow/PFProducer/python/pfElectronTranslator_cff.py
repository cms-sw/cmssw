import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFProducer.pfElectronTranslator_cfi import *
from RecoParticleFlow.PFProducer.pfElectronInterestingDetIds_cfi import *
from RecoParticleFlow.PFProducer.pfBasedElectronPhotonIso_cff import * 

pfElectronTranslatorSequence = cms.Sequence(
    pfBasedElectronPhotonIsoSequence+
    pfElectronTranslator+
    pfElectronInterestingEcalDetIdEB+
    pfElectronInterestingEcalDetIdEE
    )

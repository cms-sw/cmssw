import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFProducer.pfElectronTranslator_cfi import *
from RecoParticleFlow.PFProducer.pfElectronInterestingDetIds_cfi import *
from RecoParticleFlow.PFProducer.pfBasedElectronIso_cff import * 

pfElectronTranslatorSequence = cms.Sequence(
    pfBasedElectronIsoSequence+
    pfElectronTranslator+
    pfElectronInterestingEcalDetIdEB+
    pfElectronInterestingEcalDetIdEE
    )

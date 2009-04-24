import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFProducer.pfElectronTranslator_cfi import *
from RecoParticleFlow.PFProducer.pfElectronInterestingDetIds_cfi import *

pfElectronTranslatorSequence = cms.Sequence(
    pfElectronTranslator+
    pfElectronInterestingEcalDetIdEB+
    pfElectronInterestingEcalDetIdEE
    )

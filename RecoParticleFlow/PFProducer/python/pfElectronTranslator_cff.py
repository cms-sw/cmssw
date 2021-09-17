import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFProducer.pfElectronTranslator_cfi import *
from RecoParticleFlow.PFProducer.pfElectronInterestingDetIds_cfi import *
from RecoParticleFlow.PFProducer.pfBasedElectronPhotonIso_cff import * 

pfElectronTranslatorTask = cms.Task(
    pfBasedElectronPhotonIsoTask,
    pfElectronTranslator,
    pfElectronInterestingEcalDetIdEB,
    pfElectronInterestingEcalDetIdEE
)
pfElectronTranslatorSequence = cms.Sequence(pfElectronTranslatorTask)

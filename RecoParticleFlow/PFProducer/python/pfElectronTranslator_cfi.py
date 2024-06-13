import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfElectronTranslatorMVACut_cfi import *
from RecoParticleFlow.PFProducer.modules import PFElectronTranslator

pfElectronTranslator = PFElectronTranslator().clone(MVACutBlock = cms.PSet(pfElecMva))

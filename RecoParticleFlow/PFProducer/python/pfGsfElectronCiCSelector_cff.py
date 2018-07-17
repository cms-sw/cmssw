import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfGsfElectronCiCSelector_cfi import *

pfGsfElectronCiCSelectionSequence = cms.Sequence(
    cms.ignore(electronsWithPresel)+
    cms.ignore(electronsCiCLoose)
    )

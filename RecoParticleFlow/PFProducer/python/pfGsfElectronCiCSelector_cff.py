import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfGsfElectronCiCSelector_cfi import *

pfGsfElectronCiCSelectionSequence = cms.Sequence(
    electronsWithPresel+
    electronsCiCLoose
    )

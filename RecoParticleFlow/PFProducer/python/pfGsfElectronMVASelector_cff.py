import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfGsfElectronMVASelector_cfi import *

pfGsfElectronMVASelectionSequence = cms.Sequence(
    electronsWithPresel+
    mvaElectrons
    )



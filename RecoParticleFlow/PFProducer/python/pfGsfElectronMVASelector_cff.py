import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfGsfElectronMVASelector_cfi import *

pfGsfElectronMVASelectionSequence = cms.Sequence(
    cms.ignore(electronsWithPresel)+
    mvaElectrons
    )



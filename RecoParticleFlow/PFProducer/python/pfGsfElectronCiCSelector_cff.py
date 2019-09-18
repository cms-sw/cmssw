import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.pfGsfElectronCiCSelector_cfi import *

pfGsfElectronCiCSelectionTask = cms.Task(
    cms.ignore(electronsWithPresel),
    cms.ignore(electronsCiCLoose)
    )
pfGsfElectronCiCSelectionSequence = cms.Sequence(pfGsfElectronCiCSelectionTask)

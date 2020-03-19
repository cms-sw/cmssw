import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdMVABased_cfi import *

electronsWithPresel = cms.EDFilter("GsfElectronSelector",
                                   src = cms.InputTag("ecalDrivenGsfElectrons"),
                                   cut = cms.string("pt > 5 && ecalDrivenSeed && passingCutBasedPreselection"),
                                   )

mvaElectrons.electronTag = cms.InputTag('electronsWithPresel')

pfGsfElectronMVASelectionTask = cms.Task(
               electronsWithPresel,
               mvaElectrons    )
pfGsfElectronMVASelectionSequence = cms.Sequence(pfGsfElectronMVASelectionTask)

import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make final electrons.
# In the past, this was including the seeding, but this one is directly
# imported in the reco sequences since the integration with pflow.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores
from RecoEgamma.EgammaElectronProducers.gsfElectronCores_cfi import gsfElectronCores
from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCoresFromMultiCl_cff import ecalDrivenGsfElectronCoresFromMultiCl
from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *
gsfElectronTask = cms.Task(ecalDrivenGsfElectronCores,ecalDrivenGsfElectrons,gsfElectronCores,gsfElectrons)
gsfElectronSequence = cms.Sequence(gsfElectronTask)

gsfEcalDrivenElectronTask = cms.Task(ecalDrivenGsfElectronCores,ecalDrivenGsfElectrons)
gsfEcalDrivenElectronSequence = cms.Sequence(gsfEcalDrivenElectronTask)

_gsfEcalDrivenElectronTaskFromMultiCl = gsfEcalDrivenElectronTask.copy()
_gsfEcalDrivenElectronTaskFromMultiCl.add(cms.Task(ecalDrivenGsfElectronCoresFromMultiCl,ecalDrivenGsfElectronsFromMultiCl))
_gsfEcalDrivenElectronSequenceFromMultiCl = cms.Sequence(_gsfEcalDrivenElectronTaskFromMultiCl)


from RecoEgamma.EgammaElectronProducers.edBasedElectronIso_cff import *
from RecoEgamma.EgammaElectronProducers.pfBasedElectronIso_cff import *

electronIsoTask = cms.Task(
        edBasedElectronIsoTask,
        pfBasedElectronIsoTask
     )
electronIsoSequence = cms.Sequence(electronIsoTask)

gsfElectronMergingTask = cms.Task(electronIsoTask,gsfElectronCores,gsfElectrons)
gsfElectronMergingSequence = cms.Sequence(gsfElectronMergingTask)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
  gsfEcalDrivenElectronTask, _gsfEcalDrivenElectronTaskFromMultiCl
)

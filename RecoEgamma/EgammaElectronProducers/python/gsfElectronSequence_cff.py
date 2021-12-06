import FWCore.ParameterSet.Config as cms

#==============================================================================
# Sequence to make final electrons.
# In the past, this was including the seeding, but this one is directly
# imported in the reco sequences since the integration with pflow.
#==============================================================================

from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores
from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCoresHGC_cff import ecalDrivenGsfElectronCoresHGC
from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *

gsfEcalDrivenElectronTask = cms.Task(ecalDrivenGsfElectronCores,ecalDrivenGsfElectrons)
gsfEcalDrivenElectronSequence = cms.Sequence(gsfEcalDrivenElectronTask)

_gsfEcalDrivenElectronTaskHGC = gsfEcalDrivenElectronTask.copy()
_gsfEcalDrivenElectronTaskHGC.add(cms.Task(ecalDrivenGsfElectronCoresHGC,ecalDrivenGsfElectronsHGC))

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
  gsfEcalDrivenElectronTask, _gsfEcalDrivenElectronTaskHGC
)

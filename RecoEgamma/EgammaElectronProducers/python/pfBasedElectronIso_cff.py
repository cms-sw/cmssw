import FWCore.ParameterSet.Config as cms
 
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationDeposits_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationValues_cff import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectronFinalizer_cfi import *

# The following should be removed up to  <--- when moving to GED only
pfBasedElectronIsoTask = cms.Task(
    pfParticleSelectionTask,
    electronPFIsolationDepositsTask,
    electronPFIsolationValuesTask
    )
pfBasedElectronIsoSequence = cms.Sequence(pfBasedElectronIsoTask)
# <---- Up to here

gedElectronPFIsoTask = cms.Task(
    pfParticleSelectionTask,
    gedGsfElectrons
)
gedElectronPFIsoSequence = cms.Sequence(gedElectronPFIsoTask)


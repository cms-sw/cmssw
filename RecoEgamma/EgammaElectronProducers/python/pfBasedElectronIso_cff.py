import FWCore.ParameterSet.Config as cms
 
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationDeposits_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationValues_cff import *
from RecoEgamma.EgammaElectronProducers.gedGsfElectronFinalizer_cfi import *

# The following should be removed up to  <--- when moving to GED only
pfBasedElectronIsoSequence = cms.Sequence(
    pfParticleSelectionSequence +
    electronPFIsolationDepositsSequence +
    electronPFIsolationValuesSequence
    )
# <---- Up to here

gedElectronPFIsoSequence = cms.Sequence(
    pfParticleSelectionSequence +
    gedElectronPFIsolationDepositsSequence+
    gedElectronPFIsolationValuesSequence+
    gedGsfElectrons
)

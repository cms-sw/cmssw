import FWCore.ParameterSet.Config as cms
 
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationDeposits_cff import *
from RecoEgamma.EgammaElectronProducers.electronPFIsolationValues_cff import *


pfBasedElectronIsoSequence = cms.Sequence(
    pfParticleSelectionSequence +
    electronPFIsolationDepositsSequence +
    electronPFIsolationValuesSequence
    )



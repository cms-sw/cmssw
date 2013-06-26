import FWCore.ParameterSet.Config as cms
 
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoEgamma.EgammaElectronProducers.electronEDIsolationDeposits_cff import *
from RecoEgamma.EgammaElectronProducers.electronEDIsolationValues_cff import *


edBasedElectronIsoSequence = cms.Sequence(
    pfParticleSelectionSequence +
    electronEDIsolationDepositsSequence +
    electronEDIsolationValuesSequence
    )


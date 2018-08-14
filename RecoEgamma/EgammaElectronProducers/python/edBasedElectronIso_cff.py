import FWCore.ParameterSet.Config as cms
 
from CommonTools.ParticleFlow.pfParticleSelection_cff import *
from RecoEgamma.EgammaElectronProducers.electronEDIsolationDeposits_cff import *
from RecoEgamma.EgammaElectronProducers.electronEDIsolationValues_cff import *


edBasedElectronIsoTask = cms.Task(
    pfParticleSelectionTask,
    electronEDIsolationDepositsTask,
    electronEDIsolationValuesTask
    )
edBasedElectronIsoSequence = cms.Sequence(edBasedElectronIsoTask)

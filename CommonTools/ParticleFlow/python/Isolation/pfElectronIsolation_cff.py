import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *

pfElectronIsolationTask = cms.Task(
    electronPFIsolationDepositsTask ,
    electronPFIsolationValuesTask
    )
pfElectronIsolationSequence = cms.Sequence(pfElectronIsolationTask)

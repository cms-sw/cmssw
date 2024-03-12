import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *

pfElectronIsolationTask = cms.Task(
    electronPFIsolationDepositsTask ,
    electronPFIsolationValuesTask
    )
pfElectronIsolationSequence = cms.Sequence(pfElectronIsolationTask)
# foo bar baz
# oCXDF42H5jBb2
# nFuFaB2LR2qa1

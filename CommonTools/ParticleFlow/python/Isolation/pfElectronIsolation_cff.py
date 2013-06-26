import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.electronPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *

pfElectronIsolationSequence = cms.Sequence(
    electronPFIsolationDepositsSequence +
    electronPFIsolationValuesSequence
    )


import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFProducer.photonPFIsolationDeposits_cff import *
from RecoParticleFlow.PFProducer.photonPFIsolationValues_cff import *

pfPhotonIsolationSequence = cms.Sequence(
    photonPFIsolationDepositsSequence +
    photonPFIsolationValuesSequence
    )


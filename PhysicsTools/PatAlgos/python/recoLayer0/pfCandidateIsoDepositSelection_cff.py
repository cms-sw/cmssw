import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.sortByType_cff import *
from PhysicsTools.PFCandProducer.pfNoPileUp_cff import *


patPFCandidateIsoDepositSelection = cms.Sequence(
    pfNoPileUpSequence +
    sortByTypeSequence
    )





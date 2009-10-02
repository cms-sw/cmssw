import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.pfSortByType_cff import *
from PhysicsTools.PFCandProducer.pfNoPileUp_cff import *


patPFCandidateIsoDepositSelection = cms.Sequence(
    pfNoPileUpSequence *
    ( pfAllNeutralHadrons +
      pfAllChargedHadrons +
      pfAllPhotons )
    )





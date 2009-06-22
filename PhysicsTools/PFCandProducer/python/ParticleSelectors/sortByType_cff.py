import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.allNeutralHadrons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.allChargedHadrons_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.allPhotons_cfi import *

sortByTypeSequence = cms.Sequence(
    allNeutralHadrons+
    allChargedHadrons+
    allPhotons
    )


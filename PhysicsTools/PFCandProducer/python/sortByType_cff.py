import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.allNeutralHadrons_cfi  import *
from PhysicsTools.PFCandProducer.allChargedHadrons_cfi import *
from PhysicsTools.PFCandProducer.allPhotons_cfi import *

sortByTypeSequence = cms.Sequence(
    allNeutralHadrons+
    allChargedHadrons+
    allPhotons
    )


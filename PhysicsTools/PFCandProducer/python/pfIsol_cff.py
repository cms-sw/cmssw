import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfAllNeutralHadrons_cfi  import *
from PhysicsTools.PFCandProducer.pfAllChargedHadrons_cfi import *
from PhysicsTools.PFCandProducer.pfAllPhotons_cfi import *

pfIsol = cms.Sequence(
pfAllNeutralHadrons+
pfAllChargedHadrons+
pfAllPhotons
    )


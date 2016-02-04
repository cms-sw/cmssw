import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllNeutralHadrons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllChargedHadrons_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllPhotons_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllMuons_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllElectrons_cfi import *

pfSortByTypeSequence = cms.Sequence(
    pfAllNeutralHadrons+
    pfAllChargedHadrons+
    pfAllPhotons+
    pfAllElectrons+
    pfAllMuons
    )


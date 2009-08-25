import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllElectrons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfElectronsPtGt5_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolation_cff import *
from PhysicsTools.PFCandProducer.Isolation.pfIsolatedElectrons_cfi import *



pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    pfElectronsPtGt5 +
    # computing isolation variables:
    pfElectronIsolationSequence +
    # selecting isolated electrons:
    pfIsolatedElectrons 
    )





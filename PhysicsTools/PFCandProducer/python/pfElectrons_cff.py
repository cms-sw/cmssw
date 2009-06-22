import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.pfPileUp_cff  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllElectrons_cfi  import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfElectronsPtGt5_cfi import *
from PhysicsTools.PFCandProducer.Isolation.electronIsolation_cff import *
from PhysicsTools.PFCandProducer.ParticleSelectors.isolatedElectrons_cfi import *



pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    pfElectronsPtGt5 +
    # computing isolation variables:
    electronIsolationSequence +
    # selecting isolated electrons:
    isolatedElectrons 
    )





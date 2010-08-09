import FWCore.ParameterSet.Config as cms

from PhysicsTools.PFCandProducer.ParticleSelectors.pfAllElectrons_cfi  import *
#from PhysicsTools.PFCandProducer.ParticleSelectors.pfElectronsPtGt5_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfElectronsFromVertex_cfi import *
from PhysicsTools.PFCandProducer.ParticleSelectors.pfSelectedElectrons_cfi import *
from PhysicsTools.PFCandProducer.Isolation.pfElectronIsolation_cff import *
from PhysicsTools.PFCandProducer.Isolation.pfIsolatedElectrons_cfi import *



pfElectronSequence = cms.Sequence(
    pfAllElectrons +
    # electron selection:
    #pfElectronsPtGt5 +
    pfElectronsFromVertex +
    pfSelectedElectrons +
    # computing isolation variables:
    pfElectronIsolationSequence +
    # selecting isolated electrons:
    pfIsolatedElectrons 
    )





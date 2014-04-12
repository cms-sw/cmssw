import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.producersLayer1.photonProducer_cff import *
# (must be decleared after PAT sequence Yen-Jie)
from RecoHI.HiEgammaAlgos.HiEgammaIsolation_cff import *

makeHeavyIonPhotons = cms.Sequence(
    # reco pre-production
    hiEgammaIsolationSequence *
    patPhotonIsolation *
    # pat and HI specifics    
    photonMatch *
    # object production
    patPhotons
    )




import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitOOTECAL_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECALUncorrected_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECAL_cff import *
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterOOTECAL_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotonCore_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotons_cff import *

# task+sequence to make OOT photons from clusters in ECAL from full PFRecHits w/o timing cut
ootPhotonTask = cms.Task(
    particleFlowRecHitOOTECAL,
    particleFlowClusterOOTECALUncorrected,
    particleFlowClusterOOTECAL, 
    particleFlowSuperClusterOOTECAL, 
    ootPhotonCore, 
    ootPhotonsTmp,
    ootPhotons
    )

ootPhotonSequence = cms.Sequence(ootPhotonTask)

import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitOOTECAL_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECALUncorrected_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECAL_cff import *
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterOOTECAL_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotons_cff import *
from RecoEgamma.EgammaPhotonProducers.ootPhotonCore_cff import *

# sequence to make OOT photons from clusters in ECAL from full PFRecHits w/o timing cut

ootPhotonSequence = cms.Sequence(particleFlowOOTRecHitECAL*particleFlowClusterOOTECALUncorrected*particleFlowClusterOOTECAL*particleFlowSuperClusterOOTECAL*ootPhotonCore*ootPhotons)




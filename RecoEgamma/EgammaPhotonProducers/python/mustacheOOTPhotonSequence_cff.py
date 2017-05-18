import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitOOTECAL_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECALUncorrected_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterOOTECAL_cff import *
from RecoEcal.EgammaClusterProducers.particleFlowSuperClusterOOTECAL_cff import *
from RecoEgamma.EgammaPhotonProducers.mustacheOOTPhotons_cff import *
from RecoEgamma.EgammaPhotonProducers.mustacheOOTPhotonCore_cff import *

# sequence to make OOT photons from clusters in ECAL from full PFRecHits w/o timing cut

mustacheOOTPhotonSequence = cms.Sequence(particleFlowOOTRecHitECAL*particleFlowClusterOOTECALUncorrected*particleFlowClusterOOTECAL*particleFlowSuperClusterOOTECAL*mustacheOOTPhotonCore*mustacheOOTPhotons)




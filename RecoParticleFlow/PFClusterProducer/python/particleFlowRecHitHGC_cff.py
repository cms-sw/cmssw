from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGCEE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGCHEF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGCHEB_cfi import *

particleFlowRecHitHGC = cms.Sequence( particleFlowRecHitHGCEE  +
                                      particleFlowRecHitHGCHEF +
                                      particleFlowRecHitHGCHEB )

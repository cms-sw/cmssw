from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGCEE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGCHEF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGCHEB_cfi import *

particleFlowClusterHGC = cms.Sequence( particleFlowClusterHGCEE  +
                                       particleFlowClusterHGCHEF +
                                       particleFlowClusterHGCHEB   )
